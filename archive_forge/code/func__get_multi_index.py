import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def _get_multi_index(self, arr, indices):
    """Mimic multi dimensional indexing.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed.
        indices : tuple of index objects

        Returns
        -------
        out : ndarray
            An array equivalent to the indexing operation (but always a copy).
            `arr[indices]` should be identical.
        no_copy : bool
            Whether the indexing operation requires a copy. If this is `True`,
            `np.may_share_memory(arr, arr[indices])` should be `True` (with
            some exceptions for scalars and possibly 0-d arrays).

        Notes
        -----
        While the function may mostly match the errors of normal indexing this
        is generally not the case.
        """
    in_indices = list(indices)
    indices = []
    no_copy = True
    num_fancy = 0
    fancy_dim = 0
    error_unless_broadcast_to_empty = False
    ndim = 0
    ellipsis_pos = None
    for i, indx in enumerate(in_indices):
        if indx is None:
            continue
        if isinstance(indx, np.ndarray) and indx.dtype == bool:
            no_copy = False
            if indx.ndim == 0:
                raise IndexError
            ndim += indx.ndim
            fancy_dim += indx.ndim
            continue
        if indx is Ellipsis:
            if ellipsis_pos is None:
                ellipsis_pos = i
                continue
            raise IndexError
        if isinstance(indx, slice):
            ndim += 1
            continue
        if not isinstance(indx, np.ndarray):
            try:
                indx = np.array(indx, dtype=np.intp)
            except ValueError:
                raise IndexError
            in_indices[i] = indx
        elif indx.dtype.kind != 'b' and indx.dtype.kind != 'i':
            raise IndexError('arrays used as indices must be of integer (or boolean) type')
        if indx.ndim != 0:
            no_copy = False
        ndim += 1
        fancy_dim += 1
    if arr.ndim - ndim < 0:
        raise IndexError
    if ndim == 0 and None not in in_indices:
        return (arr.copy(), no_copy)
    if ellipsis_pos is not None:
        in_indices[ellipsis_pos:ellipsis_pos + 1] = [slice(None, None)] * (arr.ndim - ndim)
    for ax, indx in enumerate(in_indices):
        if isinstance(indx, slice):
            indx = np.arange(*indx.indices(arr.shape[ax]))
            indices.append(['s', indx])
            continue
        elif indx is None:
            indices.append(['n', np.array([0], dtype=np.intp)])
            arr = arr.reshape(arr.shape[:ax] + (1,) + arr.shape[ax:])
            continue
        if isinstance(indx, np.ndarray) and indx.dtype == bool:
            if indx.shape != arr.shape[ax:ax + indx.ndim]:
                raise IndexError
            try:
                flat_indx = np.ravel_multi_index(np.nonzero(indx), arr.shape[ax:ax + indx.ndim], mode='raise')
            except Exception:
                error_unless_broadcast_to_empty = True
                flat_indx = np.array([0] * indx.sum(), dtype=np.intp)
            if indx.ndim != 0:
                arr = arr.reshape(arr.shape[:ax] + (np.prod(arr.shape[ax:ax + indx.ndim]),) + arr.shape[ax + indx.ndim:])
                indx = flat_indx
            else:
                raise IndexError
        elif indx.ndim == 0:
            if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                raise IndexError
        if indx.ndim == 0:
            if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                raise IndexError
        if len(indices) > 0 and indices[-1][0] == 'f' and (ax != ellipsis_pos):
            indices[-1].append(indx)
        else:
            num_fancy += 1
            indices.append(['f', indx])
    if num_fancy > 1 and (not no_copy):
        new_indices = indices[:]
        axes = list(range(arr.ndim))
        fancy_axes = []
        new_indices.insert(0, ['f'])
        ni = 0
        ai = 0
        for indx in indices:
            ni += 1
            if indx[0] == 'f':
                new_indices[0].extend(indx[1:])
                del new_indices[ni]
                ni -= 1
                for ax in range(ai, ai + len(indx[1:])):
                    fancy_axes.append(ax)
                    axes.remove(ax)
            ai += len(indx) - 1
        indices = new_indices
        arr = arr.transpose(*fancy_axes + axes)
    ax = 0
    for indx in indices:
        if indx[0] == 'f':
            if len(indx) == 1:
                continue
            orig_shape = arr.shape
            orig_slice = orig_shape[ax:ax + len(indx[1:])]
            arr = arr.reshape(arr.shape[:ax] + (np.prod(orig_slice).astype(int),) + arr.shape[ax + len(indx[1:]):])
            res = np.broadcast(*indx[1:])
            if res.size != 0:
                if error_unless_broadcast_to_empty:
                    raise IndexError
                for _indx, _size in zip(indx[1:], orig_slice):
                    if _indx.size == 0:
                        continue
                    if np.any(_indx >= _size) or np.any(_indx < -_size):
                        raise IndexError
            if len(indx[1:]) == len(orig_slice):
                if np.prod(orig_slice) == 0:
                    try:
                        mi = np.ravel_multi_index(indx[1:], orig_slice, mode='raise')
                    except Exception:
                        raise IndexError('invalid index into 0-sized')
                else:
                    mi = np.ravel_multi_index(indx[1:], orig_slice, mode='wrap')
            else:
                raise ValueError
            arr = arr.take(mi.ravel(), axis=ax)
            try:
                arr = arr.reshape(arr.shape[:ax] + mi.shape + arr.shape[ax + 1:])
            except ValueError:
                raise IndexError
            ax += mi.ndim
            continue
        arr = arr.take(indx[1], axis=ax)
        ax += 1
    return (arr, no_copy)