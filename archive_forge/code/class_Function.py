import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
class Function(RObjectMixin, rinterface.SexpClosure):
    """ Python representation of an R function.
    """
    __local = baseenv_ri.find('local')
    __call = baseenv_ri.find('call')
    __assymbol = baseenv_ri.find('as.symbol')
    __newenv = baseenv_ri.find('new.env')
    _local_env = None

    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args, **kwargs)
        self._local_env = self.__newenv(hash=rinterface.BoolSexpVector((True,)))

    @docstring_property(__doc__)
    def __doc__(self) -> str:
        fm = _formals_fixed(self)
        doc = list(['Python representation of an R function.', 'R arguments:', ''])
        if fm is rinterface.NULL:
            doc.append('<No information available>')
        else:
            for key, val in zip(fm.do_slot('names'), fm):
                if key == '...':
                    val = 'R ellipsis (any number of parameters)'
                doc.append('%s: %s' % (key, _repr_argval(val)))
        return os.linesep.join(doc)

    def __call__(self, *args, **kwargs):
        cv = conversion.get_conversion()
        new_args = [cv.py2rpy(a) for a in args]
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, rinterface.Sexp):
                new_kwargs[k] = v
            else:
                new_kwargs[k] = cv.py2rpy(v)
        res = super(Function, self).__call__(*new_args, **new_kwargs)
        res = cv.rpy2py(res)
        return res

    def formals(self):
        """ Return the signature of the underlying R function
        (as the R function 'formals()' would).
        """
        res = _formals_fixed(self)
        res = conversion.get_conversion().rpy2py(res)
        return res

    def rcall(self, keyvals, environment: typing.Optional[rinterface.SexpEnvironment]=None) -> rinterface.sexp.Sexp:
        """ Wrapper around the parent method
        rpy2.rinterface.SexpClosure.rcall(). """
        res = super(Function, self).rcall(keyvals, environment=environment)
        return res