from warnings import warn
from scipy import sparse
import _scs_direct
def _select_scs_module(stgs):
    if stgs.pop('gpu', False):
        if not stgs.pop('use_indirect', _USE_INDIRECT_DEFAULT):
            raise NotImplementedError('GPU direct solver not yet available, pass `use_indirect=True`.')
        import _scs_gpu
        return _scs_gpu
    if stgs.pop('mkl', False):
        if stgs.pop('use_indirect', False):
            raise NotImplementedError('MKL indirect solver not yet available, pass `use_indirect=False`.')
        import _scs_mkl
        return _scs_mkl
    if stgs.pop('use_indirect', _USE_INDIRECT_DEFAULT):
        import _scs_indirect
        return _scs_indirect
    return _scs_direct