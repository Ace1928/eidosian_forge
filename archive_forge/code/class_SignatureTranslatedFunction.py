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
class SignatureTranslatedFunction(Function):
    """ Python representation of an R function, where
    the names in named argument are translated to valid
    argument names in Python. """
    _prm_translate: Union[OrderedDict, dict] = {}

    def __init__(self, sexp: rinterface.SexpClosure, init_prm_translate=None, on_conflict='warn', symbol_r2python=default_symbol_r2python, symbol_resolve=default_symbol_resolve):
        super(SignatureTranslatedFunction, self).__init__(sexp)
        if init_prm_translate is None:
            self._prm_translate = OrderedDict()
        else:
            assert isinstance(init_prm_translate, dict)
            self._prm_translate = init_prm_translate
        formals = self.formals()
        if formals.__sexp__._cdata != rinterface.NULL.__sexp__._cdata:
            symbol_mapping, conflicts, resolutions = _map_symbols(formals.names, translation=self._prm_translate, symbol_r2python=symbol_r2python, symbol_resolve=symbol_resolve)
            msg_prefix = "Conflict when converting R symbols in the function's signature:\n- "
            exception = ValueError
            _fix_map_symbols(symbol_mapping, conflicts, on_conflict, msg_prefix, exception)
            symbol_mapping.update(resolutions)
            self._prm_translate.update(((k, v[0]) for k, v in symbol_mapping.items()))
        if hasattr(sexp, '__rname__'):
            self.__rname__ = sexp.__rname__

    def __call__(self, *args, **kwargs):
        prm_translate = self._prm_translate
        for k in tuple(kwargs.keys()):
            r_k = prm_translate.get(k, None)
            if r_k is not None:
                v = kwargs.pop(k)
                kwargs[r_k] = v
        return super(SignatureTranslatedFunction, self).__call__(*args, **kwargs)