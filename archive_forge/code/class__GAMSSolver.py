from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
class _GAMSSolver(object):
    """Aggregate of common methods for GAMS interfaces"""

    def __init__(self, **kwds):
        self._version = None
        self._default_variable_value = None
        self._metasolver = False
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False
        self.options = Bunch()

    def version(self):
        """Returns a 4-tuple describing the solver executable version."""
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def warm_start_capable(self):
        """True is the solver can accept a warm-start solution."""
        return True

    def default_variable_value(self):
        return self._default_variable_value

    def set_options(self, istr):
        if isinstance(istr, str):
            istr = self._options_string_to_dict(istr)
        for key in istr:
            if not istr[key] is None:
                setattr(self.options, key, istr[key])

    @staticmethod
    def _options_string_to_dict(istr):
        ans = {}
        istr = istr.strip()
        if not istr:
            return ans
        if istr[0] == "'" or istr[0] == '"':
            istr = eval(istr)
        tokens = shlex.split(istr)
        for token in tokens:
            index = token.find('=')
            if index == -1:
                raise ValueError("Solver options must have the form option=value: '%s'" % istr)
            try:
                val = eval(token[index + 1:])
            except:
                val = token[index + 1:]
            ans[token[:index]] = val
        return ans

    def _simple_model(self, n):
        return '\n            option limrow = 0;\n            option limcol = 0;\n            option solprint = off;\n            set I / 1 * %s /;\n            variables ans;\n            positive variables x(I);\n            equations obj;\n            obj.. ans =g= sum(I, x(I));\n            model test / all /;\n            solve test using lp minimizing ans;\n            ' % (n,)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass