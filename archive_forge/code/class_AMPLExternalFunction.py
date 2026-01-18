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
class AMPLExternalFunction(ExternalFunction):
    __autoslot_mappers__ = {'_so': AutoSlots.encode_as_none, '_known_functions': AutoSlots.encode_as_none}

    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError('AMPLExternalFunction constructor does not support positional arguments')
        self._library = kwargs.pop('library', None)
        self._function = kwargs.pop('function', None)
        self._known_functions = None
        self._so = None
        if self._library is not None:
            _lib = find_library(self._library)
            if _lib is not None:
                self._library = _lib
            else:
                logger.warning(f'Defining AMPL external function, but cannot locate specified library "{self._library}"')
        ExternalFunction.__init__(self, *args, **kwargs)

    def _evaluate(self, args, fixed, fgh):
        if self._so is None:
            self.load_library()
        if self._function not in self._known_functions:
            raise RuntimeError("Error: external function '%s' was not registered within external library %s.\n\tAvailable functions: (%s)" % (self._function, self._library, ', '.join(self._known_functions.keys())))
        N = len(args)
        arglist = _ARGLIST(args, fgh, fixed)
        fcn = self._known_functions[self._function][0]
        f = fcn(byref(arglist))
        if fgh >= 1:
            g = [nan] * N
            for i in range(N):
                if arglist.at[i] < 0:
                    continue
                g[i] = arglist.derivs[arglist.at[i]]
        else:
            g = None
        if fgh >= 2:
            h = [nan] * ((N + N ** 2) // 2)
            for j in range(N):
                j_r = arglist.at[j]
                if j_r < 0:
                    continue
                for i in range(j + 1):
                    i_r = arglist.at[i]
                    if i_r < 0:
                        continue
                    h[i + j * (j + 1) // 2] = arglist.hes[i_r + j_r * (j_r + 1) // 2]
        else:
            h = None
        return (f, g, h)

    def load_library(self):
        _abs_lib = find_library(self._library)
        if _abs_lib is not None:
            self._library = _abs_lib
        self._so = cdll.LoadLibrary(self._library)
        self._known_functions = {}
        AE = _AMPLEXPORTS()
        AE.ASLdate = 20160307

        def addfunc(name, f, _type, nargs, funcinfo, ae):
            if not isinstance(name, str):
                name = name.decode()
            self._known_functions[str(name)] = (f, _type, nargs, funcinfo, ae)
        AE.Addfunc = _AMPLEXPORTS.ADDFUNC(addfunc)

        def addrandinit(ae, rss, v):
            rss(v, 1)
        AE.Addrandinit = _AMPLEXPORTS.ADDRANDINIT(addrandinit)

        def atreset(ae, a, b):
            logger.warning('AMPL External function: ignoring AtReset call in external library.  This may result in a memory leak or other undesirable behavior.')
        AE.AtReset = _AMPLEXPORTS.ATRESET(atreset)
        FUNCADD = CFUNCTYPE(None, POINTER(_AMPLEXPORTS))
        FUNCADD(('funcadd_ASL', self._so))(byref(AE))

    def _pprint(self):
        return ([('function', self._function), ('library', self._library), ('units', str(self._units)), ('arg_units', [str(u) for u in self._arg_units] if self._arg_units is not None else None)], (), None, None)