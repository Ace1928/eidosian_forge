from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _finalize_numpy(np, available):
    if not available:
        return
    from . import numeric_types
    numeric_types.native_types.add(np.ndarray)
    numeric_types.RegisterLogicalType(np.bool_)
    for t in (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        numeric_types.RegisterIntegerType(t)
        numeric_types._native_boolean_types.add(t)
    _floats = [np.float_, np.float16, np.float32, np.float64]
    if hasattr(np, 'float96'):
        _floats.append(np.float96)
    if hasattr(np, 'float128'):
        _floats.append(np.float128)
    for t in _floats:
        numeric_types.RegisterNumericType(t)
        numeric_types._native_boolean_types.add(t)
    _complex = [np.complex_, np.complex64, np.complex128]
    if hasattr(np, 'complex192'):
        _complex.append(np.complex192)
    if hasattr(np, 'complex256'):
        _complex.append(np.complex256)
    for t in _complex:
        numeric_types.RegisterComplexType(t)