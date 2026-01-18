import array
import contextlib
import os
import types
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.embedded
import rpy2.rinterface_lib.openrlib
import rpy2.rlike.container as rlc
from rpy2.robjects.robject import RObjectMixin, RObject
import rpy2.robjects.functions
from rpy2.robjects.environments import (Environment,
from rpy2.robjects.methods import methods_env
from rpy2.robjects.methods import RS4
from . import conversion
from . import vectors
from . import language
from rpy2.rinterface import (Sexp,
from rpy2.robjects.functions import Function
from rpy2.robjects.functions import SignatureTranslatedFunction
class Formula(RObjectMixin, rinterface.Sexp):

    def __init__(self, formula, environment=_globalenv):
        if isinstance(formula, str):
            inpackage = rinterface.baseenv['::']
            asformula = inpackage(rinterface.StrSexpVector(['stats']), rinterface.StrSexpVector(['as.formula']))
            formula = rinterface.StrSexpVector([formula])
            robj = asformula(formula, env=environment)
        else:
            robj = formula
        super(Formula, self).__init__(robj)

    def getenvironment(self):
        """ Get the environment in which the formula is finding its symbols."""
        res = self.do_slot('.Environment')
        res = conversion.get_conversion().rpy2py(res)
        return res

    def setenvironment(self, val):
        """ Set the environment in which a formula will find its symbols."""
        if not isinstance(val, rinterface.SexpEnvironment):
            raise TypeError('The environment must be an instance of rpy2.rinterface.Sexp.environment')
        self.do_slot_assign('.Environment', val)
    environment = property(getenvironment, setenvironment, None, 'R environment in which the formula will look for its variables.')