import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def _binary_operation(self, ufunc, method, *args, **kwargs):
    """Run recursion to perform binary_funcs on BlockVector"""
    x1 = args[0]
    x2 = args[1]
    if isinstance(x1, BlockVector) and isinstance(x2, BlockVector):
        assert_block_structure(x1)
        assert_block_structure(x2)
        assert x1.nblocks == x2.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
        assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(x1.size, x2.size)
        res = BlockVector(x1.nblocks)
        for i in range(x1.nblocks):
            _args = [x1.get_block(i)] + [x2.get_block(i)] + [args[j] for j in range(2, len(args))]
            res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
        return res
    elif type(x1) == np.ndarray and isinstance(x2, BlockVector):
        assert_block_structure(x2)
        assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(x1.size, x2.size)
        res = BlockVector(x2.nblocks)
        accum = 0
        for i in range(x2.nblocks):
            nelements = x2._brow_lengths[i]
            _args = [x1[accum:accum + nelements]] + [x2.get_block(i)] + [args[j] for j in range(2, len(args))]
            res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            accum += nelements
        return res
    elif type(x2) == np.ndarray and isinstance(x1, BlockVector):
        assert_block_structure(x1)
        assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(x1.size, x2.size)
        res = BlockVector(x1.nblocks)
        accum = 0
        for i in range(x1.nblocks):
            nelements = x1._brow_lengths[i]
            _args = [x1.get_block(i)] + [x2[accum:accum + nelements]] + [args[j] for j in range(2, len(args))]
            res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            accum += nelements
        return res
    elif np.isscalar(x1) and isinstance(x2, BlockVector):
        assert_block_structure(x2)
        res = BlockVector(x2.nblocks)
        for i in range(x2.nblocks):
            _args = [x1] + [x2.get_block(i)] + [args[j] for j in range(2, len(args))]
            res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
        return res
    elif np.isscalar(x2) and isinstance(x1, BlockVector):
        assert_block_structure(x1)
        res = BlockVector(x1.nblocks)
        for i in range(x1.nblocks):
            _args = [x1.get_block(i)] + [x2] + [args[j] for j in range(2, len(args))]
            res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
        return res
    elif (type(x1) == np.ndarray or np.isscalar(x1)) and (type(x2) == np.ndarray or np.isscalar(x2)):
        return super(BlockVector, self).__array_ufunc__(ufunc, method, *args, **kwargs)
    else:
        if x1.__class__.__name__ == 'MPIBlockVector':
            raise RuntimeError('Operation not supported by BlockVector')
        if x2.__class__.__name__ == 'MPIBlockVector':
            raise RuntimeError('Operation not supported by BlockVector')
        raise NotImplementedError()