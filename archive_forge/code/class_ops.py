import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
class ops(_TestParametrizer):

    def __init__(self, op_list, *, dtypes: Union[OpDTypes, Sequence[torch.dtype]]=OpDTypes.supported, allowed_dtypes: Optional[Sequence[torch.dtype]]=None):
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        """ Parameterizes the given test function across each op and its associated dtypes. """
        if device_cls is None:
            raise RuntimeError('The @ops decorator is only intended to be used in a device-specific context; use it with instantiate_device_type_tests() instead of instantiate_parametrized_tests()')
        op = check_exhausted_iterator = object()
        for op in self.op_list:
            dtypes: Union[Set[torch.dtype], Set[None]]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(op.supported_backward_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = op.supported_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.any_one:
                supported = op.supported_dtypes(device_cls.device_type)
                supported_backward = op.supported_backward_dtypes(device_cls.device_type)
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                supported = op.dtypes.intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {next((dtype for dtype in ANY_DTYPE_ORDER if dtype in supported))}
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f'Unknown OpDType: {self.opinfo_dtypes}')
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)
            test_name = op.formatted_name
            for dtype in dtypes:
                param_kwargs = {'op': op}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        try:
                            return test(*args, **kwargs)
                        except unittest.SkipTest as e:
                            raise e
                        except Exception as e:
                            tracked_input = get_tracked_input()
                            if PRINT_REPRO_ON_FAILURE and tracked_input is not None:
                                raise Exception(f'Caused by {tracked_input.type_desc} at index {tracked_input.index}: {_serialize_sample(tracked_input.val)}') from e
                            raise e
                        finally:
                            clear_tracked_input()
                    test.tracked_input = None
                    decorator_fn = partial(op.get_decorators, generic_cls.__name__, test.__name__, device_cls.device_type, dtype)
                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    print(f'Failed to instantiate {test_name} for op {op.name}!')
                    raise ex
        if op is check_exhausted_iterator:
            raise ValueError('An empty op_list was passed to @ops. Note that this may result from reuse of a generator.')