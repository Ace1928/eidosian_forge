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
class R(object):
    """
    Singleton representing the embedded R running.
    """
    _instance = None
    _print_r_warnings: bool = True
    _invisible: bool = True

    def __new__(cls):
        if cls._instance is None:
            rinterface.initr_simple()
            cls._instance = object.__new__(cls)
        return cls._instance

    def __getattribute__(self, attr: str) -> object:
        try:
            return super(R, self).__getattribute__(attr)
        except AttributeError as ae:
            orig_ae = str(ae)
        try:
            return self.__getitem__(attr)
        except LookupError:
            raise AttributeError(orig_ae)

    def __getitem__(self, item: str) -> object:
        res = _globalenv.find(item)
        res = conversion.get_conversion().rpy2py(res)
        if hasattr(res, '__rname__'):
            res.__rname__ = item
        return res

    def __cleanup__(self) -> None:
        rinterface.embedded.endr(0)
        del self

    def __str__(self) -> str:
        s = [super(R, self).__str__()]
        version = self['version']
        version_k: typing.Tuple[str, ...] = tuple(version.names)
        version_v: typing.Tuple[str, ...] = tuple((x[0] for x in version))
        for key, val in zip(version_k, version_v):
            s.extend('%s: %s' % (key, val))
        return os.linesep.join(s)

    def __call__(self, string: str, invisible: typing.Optional[bool]=None, print_r_warnings: typing.Optional[bool]=None) -> object:
        """Evaluate a string as R code.

        :param string: A string with R code
        :param invisible: evaluate the R expression handling R's
          invisibility flag. When `True` expressions meant to return
          an "invisible" result (for example, `x <- 1`) will return
          None. The default is `None`, in which case the attribute
        _invisible is used.
        :param print_r_warning: When `True` the R deferred warnings
          are printed using the R callback function. The default is
          `None`, in which case the attribute _print_r_warning
          is used.
        :return: The value returned by R after rpy2 conversion."""
        r_expr = rinterface.parse(string)
        if invisible is None:
            invisible = self._invisible
        if invisible:
            res, visible = rinterface.evalr_expr_with_visible(r_expr)
            if not visible[0]:
                res = None
        else:
            res = rinterface.evalr_expr(r_expr)
        if print_r_warnings is None:
            print_r_warnings = self._print_r_warnings
        if print_r_warnings:
            _print_deferred_warnings()
        return None if res is None else conversion.get_conversion().rpy2py(res)