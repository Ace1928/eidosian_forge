import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
def _check_hipsparse_generic_available():
    if not TEST_WITH_ROCM:
        return False
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split('-')[0]
    rocm_version_tuple = tuple((int(x) for x in rocm_version.split('.')))
    return not (rocm_version_tuple is None or rocm_version_tuple < (5, 1))