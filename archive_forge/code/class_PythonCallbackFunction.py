import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
class PythonCallbackFunction(ExternalFunction):
    __autoslot_mappers__ = {'_fcn_id': _python_callback_fid_mapper}
    global_registry = []
    global_id_to_fid = {}

    @classmethod
    def register_instance(cls, instance):
        _id = id(instance)
        if _id in cls.global_id_to_fid:
            prev_fid = cls.global_id_to_fid[_id]
            prev_obj = cls.global_registry[prev_fid]
            if prev_obj() is not None:
                assert prev_obj() is instance
                return prev_fid
        cls.global_id_to_fid[_id] = _fid = len(cls.global_registry)
        cls.global_registry.append(weakref.ref(instance))
        return _fid

    def __init__(self, *args, **kwargs):
        for i, kw in enumerate(('function', 'gradient', 'hessian')):
            if len(args) <= i:
                break
            if kw in kwargs:
                raise ValueError(f"Duplicate definition of external function through positional and keyword ('{kw}=') arguments")
            kwargs[kw] = args[i]
        if len(args) > 3:
            raise ValueError('PythonCallbackFunction constructor only supports 0 - 3 positional arguments')
        self._fcn = kwargs.pop('function', None)
        self._grad = kwargs.pop('gradient', None)
        self._hess = kwargs.pop('hessian', None)
        self._fgh = kwargs.pop('fgh', None)
        if self._fgh is not None and any((self._fcn, self._grad, self._hess)):
            raise ValueError("Cannot specify 'fgh' with any of {'function', 'gradient', hessian'}")
        arg_units = kwargs.get('arg_units', None)
        if arg_units is not None:
            kwargs['arg_units'] = list(arg_units)
            kwargs['arg_units'].append(None)
        self._library = 'pyomo_ampl.so'
        self._function = 'pyomo_socket_server'
        ExternalFunction.__init__(self, *args, **kwargs)
        self._fcn_id = PythonCallbackFunction.register_instance(self)

    def __call__(self, *args):
        return super().__call__(*args, _PythonCallbackFunctionID(self._fcn_id))

    def _evaluate(self, args, fixed, fgh):
        _id = args.pop()
        if fixed is not None:
            fixed = fixed[:-1]
        if _id != self._fcn_id:
            raise RuntimeError('PythonCallbackFunction called with invalid Global ID')
        if self._fgh is not None:
            f, g, h = self._fgh(args, fgh, fixed)
        else:
            f = self._fcn(*args)
            if fgh >= 1:
                if self._grad is None:
                    raise RuntimeError(f"ExternalFunction '{self.name}' was not defined with a gradient callback.  Cannot evaluate the derivative of the function")
                g = self._grad(args, fixed)
            else:
                g = None
            if fgh == 2:
                if self._hess is None:
                    raise RuntimeError(f"ExternalFunction '{self.name}' was not defined with a Hessian callback.  Cannot evaluate the second derivative of the function")
                h = self._hess(args, fixed)
            else:
                h = None
        if g is not None:
            g.append(0)
        if h is not None:
            h.extend([0] * (len(args) + 1))
        return (f, g, h)

    def _pprint(self):
        return ([('function', self._fcn.__qualname__), ('units', str(self._units)), ('arg_units', [str(u) for u in self._arg_units[:-1]] if self._arg_units is not None else None)], (), None, None)