from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
class TestBlockManager:

    def test_attrs(self):
        mgr = create_mgr('a,b,c: f8-1; d,e,f: f8-2')
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    def test_duplicate_ref_loc_failure(self):
        tmp_mgr = create_mgr('a:bool; a: f8')
        axes, blocks = (tmp_mgr.axes, tmp_mgr.blocks)
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))
        msg = 'Gaps in blk ref_locs'
        with pytest.raises(AssertionError, match=msg):
            mgr = BlockManager(blocks, axes)
            mgr._rebuild_blknos_and_blklocs()
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        mgr.iget(1)

    def test_pickle(self, mgr):
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        assert hasattr(mgr2, '_is_consolidated')
        assert hasattr(mgr2, '_known_consolidated')
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    @pytest.mark.parametrize('mgr_string', ['a,a,a:f8', 'a: f8; a: i8'])
    def test_non_unique_pickle(self, mgr_string):
        mgr = create_mgr(mgr_string)
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))

    def test_categorical_block_pickle(self):
        mgr = create_mgr('a: category')
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        smgr = create_single_mgr('category')
        smgr2 = tm.round_trip_pickle(smgr)
        tm.assert_series_equal(Series()._constructor_from_mgr(smgr, axes=smgr.axes), Series()._constructor_from_mgr(smgr2, axes=smgr2.axes))

    def test_iget(self):
        cols = Index(list('abc'))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(values=values.copy(), placement=BlockPlacement(np.arange(3, dtype=np.intp)), ndim=values.ndim)
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])
        tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
        tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
        tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])

    def test_set(self):
        mgr = create_mgr('a,b,c: int', item_shape=(3,))
        mgr.insert(len(mgr.items), 'd', np.array(['foo'] * 3))
        mgr.iset(1, np.array(['bar'] * 3))
        tm.assert_numpy_array_equal(mgr.iget(0).internal_values(), np.array([0] * 3))
        tm.assert_numpy_array_equal(mgr.iget(1).internal_values(), np.array(['bar'] * 3, dtype=np.object_))
        tm.assert_numpy_array_equal(mgr.iget(2).internal_values(), np.array([2] * 3))
        tm.assert_numpy_array_equal(mgr.iget(3).internal_values(), np.array(['foo'] * 3, dtype=np.object_))

    def test_set_change_dtype(self, mgr):
        mgr.insert(len(mgr.items), 'baz', np.zeros(N, dtype=bool))
        mgr.iset(mgr.items.get_loc('baz'), np.repeat('foo', N))
        idx = mgr.items.get_loc('baz')
        assert mgr.iget(idx).dtype == np.object_
        mgr2 = mgr.consolidate()
        mgr2.iset(mgr2.items.get_loc('baz'), np.repeat('foo', N))
        idx = mgr2.items.get_loc('baz')
        assert mgr2.iget(idx).dtype == np.object_
        mgr2.insert(len(mgr2.items), 'quux', np.random.default_rng(2).standard_normal(N).astype(int))
        idx = mgr2.items.get_loc('quux')
        assert mgr2.iget(idx).dtype == np.dtype(int)
        mgr2.iset(mgr2.items.get_loc('quux'), np.random.default_rng(2).standard_normal(N))
        assert mgr2.iget(idx).dtype == np.float64

    def test_copy(self, mgr):
        cp = mgr.copy(deep=False)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            tm.assert_equal(cp_blk.values, blk.values)
            if isinstance(blk.values, np.ndarray):
                assert cp_blk.values.base is blk.values.base
            else:
                assert cp_blk.values._ndarray.base is blk.values._ndarray.base
        mgr._consolidate_inplace()
        cp = mgr.copy(deep=True)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            bvals = blk.values
            cpvals = cp_blk.values
            tm.assert_equal(cpvals, bvals)
            if isinstance(cpvals, np.ndarray):
                lbase = cpvals.base
                rbase = bvals.base
            else:
                lbase = cpvals._ndarray.base
                rbase = bvals._ndarray.base
            if isinstance(cpvals, DatetimeArray):
                assert lbase is None and rbase is None or lbase is not rbase
            elif not isinstance(cpvals, np.ndarray):
                assert lbase is not rbase
            else:
                assert lbase is None and rbase is None

    def test_sparse(self):
        mgr = create_mgr('a: sparse-1; b: sparse-2')
        assert mgr.as_array().dtype == np.float64

    def test_sparse_mixed(self):
        mgr = create_mgr('a: sparse-1; b: sparse-2; c: f8')
        assert len(mgr.blocks) == 3
        assert isinstance(mgr, BlockManager)

    @pytest.mark.parametrize('mgr_string, dtype', [('c: f4; d: f2', np.float32), ('c: f4; d: f2; e: f8', np.float64)])
    def test_as_array_float(self, mgr_string, dtype):
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize('mgr_string, dtype', [('a: bool-1; b: bool-2', np.bool_), ('a: i8-1; b: i8-2; c: i4; d: i2; e: u1', np.int64), ('c: i4; d: i2; e: u1', np.int32)])
    def test_as_array_int_bool(self, mgr_string, dtype):
        mgr = create_mgr(mgr_string)
        assert mgr.as_array().dtype == dtype

    def test_as_array_datetime(self):
        mgr = create_mgr('h: datetime-1; g: datetime-2')
        assert mgr.as_array().dtype == 'M8[ns]'

    def test_as_array_datetime_tz(self):
        mgr = create_mgr('h: M8[ns, US/Eastern]; g: M8[ns, CET]')
        assert mgr.iget(0).dtype == 'datetime64[ns, US/Eastern]'
        assert mgr.iget(1).dtype == 'datetime64[ns, CET]'
        assert mgr.as_array().dtype == 'object'

    @pytest.mark.parametrize('t', ['float16', 'float32', 'float64', 'int32', 'int64'])
    def test_astype(self, t):
        mgr = create_mgr('c: f4; d: f2; e: f8')
        t = np.dtype(t)
        tmgr = mgr.astype(t)
        assert tmgr.iget(0).dtype.type == t
        assert tmgr.iget(1).dtype.type == t
        assert tmgr.iget(2).dtype.type == t
        mgr = create_mgr('a,b: object; c: bool; d: datetime; e: f4; f: f2; g: f8')
        t = np.dtype(t)
        tmgr = mgr.astype(t, errors='ignore')
        assert tmgr.iget(2).dtype.type == t
        assert tmgr.iget(4).dtype.type == t
        assert tmgr.iget(5).dtype.type == t
        assert tmgr.iget(6).dtype.type == t
        assert tmgr.iget(0).dtype.type == np.object_
        assert tmgr.iget(1).dtype.type == np.object_
        if t != np.int64:
            assert tmgr.iget(3).dtype.type == np.datetime64
        else:
            assert tmgr.iget(3).dtype.type == t

    def test_convert(self, using_infer_string):

        def _compare(old_mgr, new_mgr):
            """compare the blocks, numeric compare ==, object don't"""
            old_blocks = set(old_mgr.blocks)
            new_blocks = set(new_mgr.blocks)
            assert len(old_blocks) == len(new_blocks)
            for b in old_blocks:
                found = False
                for nb in new_blocks:
                    if (b.values == nb.values).all():
                        found = True
                        break
                assert found
            for b in new_blocks:
                found = False
                for ob in old_blocks:
                    if (b.values == ob.values).all():
                        found = True
                        break
                assert found
        mgr = create_mgr('f: i8; g: f8')
        new_mgr = mgr.convert(copy=True)
        _compare(mgr, new_mgr)
        mgr = create_mgr('a,b,foo: object; f: i8; g: f8')
        mgr.iset(0, np.array(['1'] * N, dtype=np.object_))
        mgr.iset(1, np.array(['2.'] * N, dtype=np.object_))
        mgr.iset(2, np.array(['foo.'] * N, dtype=np.object_))
        new_mgr = mgr.convert(copy=True)
        dtype = 'string[pyarrow_numpy]' if using_infer_string else np.object_
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int64
        assert new_mgr.iget(4).dtype == np.float64
        mgr = create_mgr('a,b,foo: object; f: i4; bool: bool; dt: datetime; i: i8; g: f8; h: f2')
        mgr.iset(0, np.array(['1'] * N, dtype=np.object_))
        mgr.iset(1, np.array(['2.'] * N, dtype=np.object_))
        mgr.iset(2, np.array(['foo.'] * N, dtype=np.object_))
        new_mgr = mgr.convert(copy=True)
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int32
        assert new_mgr.iget(4).dtype == np.bool_
        assert new_mgr.iget(5).dtype.type, np.datetime64
        assert new_mgr.iget(6).dtype == np.int64
        assert new_mgr.iget(7).dtype == np.float64
        assert new_mgr.iget(8).dtype == np.float16

    def test_interleave(self):
        for dtype in ['f8', 'i8', 'object', 'bool', 'complex', 'M8[ns]', 'm8[ns]']:
            mgr = create_mgr(f'a: {dtype}')
            assert mgr.as_array().dtype == dtype
            mgr = create_mgr(f'a: {dtype}; b: {dtype}')
            assert mgr.as_array().dtype == dtype

    @pytest.mark.parametrize('mgr_string, dtype', [('a: category', 'i8'), ('a: category; b: category', 'i8'), ('a: category; b: category2', 'object'), ('a: category2', 'object'), ('a: category2; b: category2', 'object'), ('a: f8', 'f8'), ('a: f8; b: i8', 'f8'), ('a: f4; b: i8', 'f8'), ('a: f4; b: i8; d: object', 'object'), ('a: bool; b: i8', 'object'), ('a: complex', 'complex'), ('a: f8; b: category', 'object'), ('a: M8[ns]; b: category', 'object'), ('a: M8[ns]; b: bool', 'object'), ('a: M8[ns]; b: i8', 'object'), ('a: m8[ns]; b: bool', 'object'), ('a: m8[ns]; b: i8', 'object'), ('a: M8[ns]; b: m8[ns]', 'object')])
    def test_interleave_dtype(self, mgr_string, dtype):
        mgr = create_mgr('a: category')
        assert mgr.as_array().dtype == 'i8'
        mgr = create_mgr('a: category; b: category2')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: category2')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: f8')
        assert mgr.as_array().dtype == 'f8'
        mgr = create_mgr('a: f8; b: i8')
        assert mgr.as_array().dtype == 'f8'
        mgr = create_mgr('a: f4; b: i8')
        assert mgr.as_array().dtype == 'f8'
        mgr = create_mgr('a: f4; b: i8; d: object')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: bool; b: i8')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: complex')
        assert mgr.as_array().dtype == 'complex'
        mgr = create_mgr('a: f8; b: category')
        assert mgr.as_array().dtype == 'f8'
        mgr = create_mgr('a: M8[ns]; b: category')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: M8[ns]; b: bool')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: M8[ns]; b: i8')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: m8[ns]; b: bool')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: m8[ns]; b: i8')
        assert mgr.as_array().dtype == 'object'
        mgr = create_mgr('a: M8[ns]; b: m8[ns]')
        assert mgr.as_array().dtype == 'object'

    def test_consolidate_ordering_issues(self, mgr):
        mgr.iset(mgr.items.get_loc('f'), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc('d'), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc('b'), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc('g'), np.random.default_rng(2).standard_normal(N))
        mgr.iset(mgr.items.get_loc('h'), np.random.default_rng(2).standard_normal(N))
        cons = mgr.consolidate()
        assert cons.nblocks == 4
        cons = mgr.consolidate().get_numeric_data()
        assert cons.nblocks == 1
        assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(cons.blocks[0].mgr_locs.as_array, np.arange(len(cons.items), dtype=np.intp))

    def test_reindex_items(self):
        mgr = create_mgr('a: f8; b: i8; c: f8; d: i8; e: f8; f: bool; g: f8-2')
        reindexed = mgr.reindex_axis(['g', 'c', 'a', 'd'], axis=0)
        assert not reindexed.is_consolidated()
        tm.assert_index_equal(reindexed.items, Index(['g', 'c', 'a', 'd']))
        tm.assert_almost_equal(mgr.iget(6).internal_values(), reindexed.iget(0).internal_values())
        tm.assert_almost_equal(mgr.iget(2).internal_values(), reindexed.iget(1).internal_values())
        tm.assert_almost_equal(mgr.iget(0).internal_values(), reindexed.iget(2).internal_values())
        tm.assert_almost_equal(mgr.iget(3).internal_values(), reindexed.iget(3).internal_values())

    def test_get_numeric_data(self, using_copy_on_write):
        mgr = create_mgr('int: int; float: float; complex: complex;str: object; bool: bool; obj: object; dt: datetime', item_shape=(3,))
        mgr.iset(5, np.array([1, 2, 3], dtype=np.object_))
        numeric = mgr.get_numeric_data()
        tm.assert_index_equal(numeric.items, Index(['int', 'float', 'complex', 'bool']))
        tm.assert_almost_equal(mgr.iget(mgr.items.get_loc('float')).internal_values(), numeric.iget(numeric.items.get_loc('float')).internal_values())
        numeric.iset(numeric.items.get_loc('float'), np.array([100.0, 200.0, 300.0]), inplace=True)
        if using_copy_on_write:
            tm.assert_almost_equal(mgr.iget(mgr.items.get_loc('float')).internal_values(), np.array([1.0, 1.0, 1.0]))
        else:
            tm.assert_almost_equal(mgr.iget(mgr.items.get_loc('float')).internal_values(), np.array([100.0, 200.0, 300.0]))

    def test_get_bool_data(self, using_copy_on_write):
        mgr = create_mgr('int: int; float: float; complex: complex;str: object; bool: bool; obj: object; dt: datetime', item_shape=(3,))
        mgr.iset(6, np.array([True, False, True], dtype=np.object_))
        bools = mgr.get_bool_data()
        tm.assert_index_equal(bools.items, Index(['bool']))
        tm.assert_almost_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), bools.iget(bools.items.get_loc('bool')).internal_values())
        bools.iset(0, np.array([True, False, True]), inplace=True)
        if using_copy_on_write:
            tm.assert_numpy_array_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), np.array([True, True, True]))
        else:
            tm.assert_numpy_array_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), np.array([True, False, True]))

    def test_unicode_repr_doesnt_raise(self):
        repr(create_mgr('b,א: object'))

    @pytest.mark.parametrize('mgr_string', ['a,b,c: i8-1; d,e,f: i8-2', 'a,a,a: i8-1; b,b,b: i8-2'])
    def test_equals(self, mgr_string):
        bm1 = create_mgr(mgr_string)
        bm2 = BlockManager(bm1.blocks[::-1], bm1.axes)
        assert bm1.equals(bm2)

    @pytest.mark.parametrize('mgr_string', ['a:i8;b:f8', 'a:i8;b:f8;c:c8;d:b', 'a:i8;e:dt;f:td;g:string', 'a:i8;b:category;c:category2', 'c:sparse;d:sparse_na;b:f8'])
    def test_equals_block_order_different_dtypes(self, mgr_string):
        bm = create_mgr(mgr_string)
        block_perms = itertools.permutations(bm.blocks)
        for bm_perm in block_perms:
            bm_this = BlockManager(tuple(bm_perm), bm.axes)
            assert bm.equals(bm_this)
            assert bm_this.equals(bm)

    def test_single_mgr_ctor(self):
        mgr = create_single_mgr('f8', num_rows=5)
        assert mgr.external_values().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        bm1 = create_mgr('a,b,c: i8-1; d,e,f: i8-2')
        msg = f'For argument "inplace" expected type bool, received type {type(value).__name__}.'
        with pytest.raises(ValueError, match=msg):
            bm1.replace_list([1], [2], inplace=value)

    def test_iset_split_block(self):
        bm = create_mgr('a,b,c: i8; d: f8')
        bm._iset_split_block(0, np.array([0]))
        tm.assert_numpy_array_equal(bm.blklocs, np.array([0, 0, 1, 0], dtype='int64' if IS64 else 'int32'))
        tm.assert_numpy_array_equal(bm.blknos, np.array([0, 0, 0, 1], dtype='int64' if IS64 else 'int32'))
        assert len(bm.blocks) == 2

    def test_iset_split_block_values(self):
        bm = create_mgr('a,b,c: i8; d: f8')
        bm._iset_split_block(0, np.array([0]), np.array([list(range(10))]))
        tm.assert_numpy_array_equal(bm.blklocs, np.array([0, 0, 1, 0], dtype='int64' if IS64 else 'int32'))
        tm.assert_numpy_array_equal(bm.blknos, np.array([0, 2, 2, 1], dtype='int64' if IS64 else 'int32'))
        assert len(bm.blocks) == 3