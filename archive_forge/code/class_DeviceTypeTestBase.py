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
class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'
    _stop_test_suite = False
    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    @property
    def rel_tol(self):
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec

    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    @classmethod
    def _init_and_get_primary_device(cls):
        try:
            return cls.get_primary_device()
        except Exception:
            if hasattr(cls, 'setUpClass'):
                cls.setUpClass()
            return cls.get_primary_device()

    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        default_dtypes = test.dtypes.get('all')
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        assert default_dtypes is not None, msg
        return test.dtypes.get(cls.device_type, default_dtypes)

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, 'tolerance_overrides'):
            return (self.precision, self.rel_tol)
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        dtype = param_kwargs['dtype'] if 'dtype' in param_kwargs else None
        dtype = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else dtype
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []):
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if 'device' in test_sig_params or 'devices' in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, 'num_required_devices'):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, 'device', device_arg)
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    self._stop_test_suite = self._should_stop_test_suite()
                    if getattr(test, '__unittest_expecting_failure__', False) and self._stop_test_suite:
                        import sys
                        print('Suppressing fatal exception to trigger unexpected success', file=sys.stderr)
                        return
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol
                return result
            assert not hasattr(cls, name), f'Redefinition of test {name}'
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            yield (test, '', {}, lambda _: [])
        parametrize_fn = getattr(test, 'parametrize_fn', default_parametrize_fn)
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                for dtype in dtypes:
                    param_kwargs: Dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, 'dtype', dtype)
                    yield (test, '', param_kwargs, lambda _: [])
            parametrize_fn = compose_parametrize_fns(dtype_parametrize_fn, parametrize_fn)
        for test, test_suffix, param_kwargs, decorator_fn in parametrize_fn(test, generic_cls, cls):
            test_suffix = '' if test_suffix == '' else '_' + test_suffix
            device_suffix = '_' + cls.device_type
            dtype_kwarg = None
            if 'dtype' in param_kwargs or 'dtypes' in param_kwargs:
                dtype_kwarg = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else param_kwargs['dtype']
            test_name = f'{name}{test_suffix}{device_suffix}{_dtype_test_suffix(dtype_kwarg)}'
            instantiate_test_helper(cls=cls, name=test_name, test=test, param_kwargs=param_kwargs, decorator_fn=decorator_fn)

    def run(self, result=None):
        super().run(result=result)
        if self._stop_test_suite:
            result.stop()