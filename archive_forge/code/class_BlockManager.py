from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """
    ndim = 2

    def __init__(self, blocks: Sequence[Block], axes: Sequence[Index], verify_integrity: bool=True) -> None:
        if verify_integrity:
            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(f'Number of Block dimensions ({block.ndim}) must equal number of axes ({self.ndim})')
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        mgr_shape = self.shape
        tot_items = sum((len(x.mgr_locs) for x in self.blocks))
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(tot_items, block.shape[1:], self.axes)
        if len(self.items) != tot_items:
            raise AssertionError(f'Number of manager items must equal union of block items\n# manager items: {len(self.items)}, # tot_items: {tot_items}')

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        return cls(blocks, axes, verify_integrity=False)

    def fast_xs(self, loc: int) -> SingleBlockManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if len(self.blocks) == 1:
            result = self.blocks[0].iget((slice(None), loc))
            bp = BlockPlacement(slice(0, len(result)))
            block = new_block(result, placement=bp, ndim=1, refs=self.blocks[0].refs)
            return SingleBlockManager(block, self.axes[0])
        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
        n = len(self)
        if isinstance(dtype, ExtensionDtype):
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
            result = ensure_wrapped_if_datetimelike(result)
        for blk in self.blocks:
            for i, rl in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))
        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        return SingleBlockManager(block, self.axes[0])

    def iget(self, i: int, track_ref: bool=True) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        bp = BlockPlacement(slice(0, len(values)))
        nb = type(block)(values, placement=bp, ndim=1, refs=block.refs if track_ref else None)
        return SingleBlockManager(nb, self.axes[1])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> list[np.ndarray]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        result: list[np.ndarray | None] = [None] * len(self.items)
        for blk in self.blocks:
            mgr_locs = blk._mgr_locs
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                result[mgr_locs[0]] = values
            else:
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]
        return result

    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool=False, refs: BlockValuesRefs | None=None) -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """
        if self._blklocs is None and self.ndim > 1:
            self._rebuild_blknos_and_blklocs()
        value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
        if not value_is_extension_type:
            if value.ndim == 2:
                value = value.T
            else:
                value = ensure_block_shape(value, ndim=2)
            if value.shape[1:] != self.shape[1:]:
                raise AssertionError('Shape of new values must be compatible with manager shape')
        if lib.is_integer(loc):
            loc = cast(int, loc)
            blkno = self.blknos[loc]
            blk = self.blocks[blkno]
            if len(blk._mgr_locs) == 1:
                return self._iset_single(loc, value, inplace=inplace, blkno=blkno, blk=blk, refs=refs)
            loc = [loc]
        if value_is_extension_type:

            def value_getitem(placement):
                return value
        else:

            def value_getitem(placement):
                return value[placement.indexer]
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()
        unfit_mgr_locs = []
        unfit_val_locs = []
        removed_blknos = []
        for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno_l]
            blk_locs = blklocs[val_locs.indexer]
            if inplace and blk.should_store(value):
                if using_copy_on_write() and (not self._has_no_reference_block(blkno_l)):
                    self._iset_split_block(blkno_l, blk_locs, value_getitem(val_locs), refs=refs)
                else:
                    blk.set_inplace(blk_locs, value_getitem(val_locs))
                    continue
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)
                if len(val_locs) == len(blk.mgr_locs):
                    removed_blknos.append(blkno_l)
                    continue
                else:
                    self._iset_split_block(blkno_l, blk_locs, refs=refs)
        if len(removed_blknos):
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True
            new_blknos = np.empty(self.nblocks, dtype=np.intp)
            new_blknos.fill(-1)
            new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple((blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)))
        if unfit_val_locs:
            unfit_idxr = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_idxr)
            new_blocks: list[Block] = []
            if value_is_extension_type:
                new_blocks.extend((new_block_2d(values=value, placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)), refs=refs) for mgr_loc in unfit_idxr))
                self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
                self._blklocs[unfit_idxr] = 0
            else:
                unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])
                new_blocks.append(new_block_2d(values=value_getitem(unfit_val_items), placement=BlockPlacement(unfit_idxr), refs=refs))
                self._blknos[unfit_idxr] = len(self.blocks)
                self._blklocs[unfit_idxr] = np.arange(unfit_count)
            self.blocks += tuple(new_blocks)
            self._known_consolidated = False

    def _iset_split_block(self, blkno_l: int, blk_locs: np.ndarray | list[int], value: ArrayLike | None=None, refs: BlockValuesRefs | None=None) -> None:
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determinint the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        blk = self.blocks[blkno_l]
        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()
        nbs_tup = tuple(blk.delete(blk_locs))
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])
        nr_blocks = len(self.blocks)
        blocks_tup = self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1:] + nbs_tup
        self.blocks = blocks_tup
        if not nbs_tup and value is not None:
            return
        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))
        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(self, loc: int, value: ArrayLike, inplace: bool, blkno: int, blk: Block, refs: BlockValuesRefs | None=None) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        if inplace and blk.should_store(value):
            copy = False
            if using_copy_on_write() and (not self._has_no_reference_block(blkno)):
                copy = True
            iloc = self.blklocs[loc]
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return
        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1:]
        self.blocks = new_blocks
        return

    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool=False) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        needs_to_warn = False
        if warn_copy_on_write() and (not self._has_no_reference(loc)):
            if not isinstance(self.blocks[self.blknos[loc]].values, (ArrowExtensionArray, ArrowStringArray)):
                needs_to_warn = True
        elif using_copy_on_write() and (not self._has_no_reference(loc)):
            blkno = self.blknos[loc]
            blk_loc = self.blklocs[loc]
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                values = values[[blk_loc]]
            self._iset_split_block(blkno, [blk_loc], values)
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            col_mgr.setitem_inplace(idx, value)
        else:
            new_mgr = col_mgr.setitem((idx,), value)
            self.iset(loc, new_mgr._block.values, inplace=True)
        if needs_to_warn:
            warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The behavior of Index.insert with object-dtype is deprecated', category=FutureWarning)
            new_axis = self.items.insert(loc, item)
        if value.ndim == 2:
            value = value.T
            if len(value) > 1:
                raise ValueError(f'Expected a 1D array, got an array with shape {value.T.shape}')
        else:
            value = ensure_block_shape(value, ndim=self.ndim)
        bp = BlockPlacement(slice(loc, loc + 1))
        block = new_block_2d(values=value, placement=bp, refs=refs)
        if not len(self.blocks):
            self._blklocs = np.array([0], dtype=np.intp)
            self._blknos = np.array([0], dtype=np.intp)
        else:
            self._insert_update_mgr_locs(loc)
            self._insert_update_blklocs_and_blknos(loc)
        self.axes[0] = new_axis
        self.blocks += (block,)
        self._known_consolidated = False
        if sum((not block.is_extension for block in self.blocks)) > 100:
            warnings.warn('DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`', PerformanceWarning, stacklevel=find_stack_level())

    def _insert_update_mgr_locs(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
        for blkno, count in _fast_count_smallints(self.blknos[loc:]):
            blk = self.blocks[blkno]
            blk._mgr_locs = blk._mgr_locs.increment_above(loc)

    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """
        if loc == self.blklocs.shape[0]:
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            self._blklocs = np.append(self._blklocs[::-1], 0)[::-1]
            self._blknos = np.append(self._blknos[::-1], len(self.blocks))[::-1]
        else:
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(self.blklocs, self.blknos, loc, len(self.blocks))
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        taker = (~is_deleted).nonzero()[0]
        nbs = self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        new_columns = self.items[~is_deleted]
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        result_blocks: list[Block] = []
        for blk in self.blocks:
            if blk.is_object:
                for sb in blk._split():
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)
        if len(result_blocks) == 0:
            nrows = 0
        else:
            nrows = result_blocks[0].values.shape[-1]
        index = Index(range(nrows))
        return type(self).from_blocks(result_blocks, [self.axes[0], index])

    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
        assert self.ndim == 2
        res_blocks: list[Block] = []
        for blk in self.blocks:
            nbs = blk.reduce(func)
            res_blocks.extend(nbs)
        index = Index([None])
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(self, *, qs: Index, interpolation: QuantileInterpolation='linear') -> Self:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        assert self.ndim >= 2
        assert is_list_like(qs)
        new_axes = list(self.axes)
        new_axes[1] = Index(qs, dtype=np.float64)
        blocks = [blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks]
        return type(self)(blocks, new_axes)

    def unstack(self, unstacker, fill_value) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index
        allow_fill = not unstacker.mask_all
        if allow_fill:
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        else:
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)
        new_blocks: list[Block] = []
        columns_mask: list[np.ndarray] = []
        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)
            factor = int(fac)
        for blk in self.blocks:
            mgr_locs = blk.mgr_locs
            new_placement = mgr_locs.tile_for_unstack(factor)
            blocks, mask = blk._unstack(unstacker, fill_value, new_placement=new_placement, needs_masking=needs_masking)
            new_blocks.extend(blocks)
            columns_mask.extend(mask)
            assert mask.sum() == sum((len(nb._mgr_locs) for nb in blocks))
        new_columns = new_columns[columns_mask]
        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_dict(self) -> dict[str, Self]:
        """
        Return a dict of str(dtype) -> BlockManager

        Returns
        -------
        values : a dict of dtype -> BlockManager
        """
        bd: dict[str, list[Block]] = {}
        for b in self.blocks:
            bd.setdefault(str(b.dtype), []).append(b)
        return {dtype: self._combine(blocks) for dtype, blocks in bd.items()}

    def as_array(self, dtype: np.dtype | None=None, copy: bool=False, na_value: object=lib.no_default) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        passed_nan = lib.is_float(na_value) and isna(na_value)
        if len(self.blocks) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose()
        if self.is_single_block:
            blk = self.blocks[0]
            if na_value is not lib.no_default:
                if lib.is_np_dtype(blk.dtype, 'f') and passed_nan:
                    pass
                else:
                    copy = True
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value, copy=copy).reshape(blk.shape)
            else:
                arr = np.array(blk.values, dtype=dtype, copy=copy)
            if using_copy_on_write() and (not copy):
                arr = arr.view()
                arr.flags.writeable = False
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
        if na_value is lib.no_default:
            pass
        elif arr.dtype.kind == 'f' and passed_nan:
            pass
        else:
            arr[isna(arr)] = na_value
        return arr.transpose()

    def _interleave(self, dtype: np.dtype | None=None, na_value: object=lib.no_default) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            dtype = interleaved_dtype([blk.dtype for blk in self.blocks])
        dtype = ensure_np_dtype(dtype)
        result = np.empty(self.shape, dtype=dtype)
        itemmask = np.zeros(self.shape[0])
        if dtype == np.dtype('object') and na_value is lib.no_default:
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result
        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                arr = blk.values.to_numpy(dtype=dtype, na_value=na_value)
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1
        if not itemmask.all():
            raise AssertionError('Some items were not contained in blocks')
        return result

    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        if len(self.blocks) == 1:
            self._is_consolidated = True
            self._known_consolidated = True
            return
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def _consolidate_inplace(self) -> None:
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        offset = 0
        blocks: list[Block] = []
        for mgr in mgrs:
            for blk in mgr.blocks:
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                blocks.append(nb)
            offset += len(mgr.items)
        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        raise NotImplementedError('This logic lives (for now) in internals.concat')