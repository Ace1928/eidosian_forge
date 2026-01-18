import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
class RObject(RObjectMixin, rpy2.rinterface.Sexp):
    """ Base class for all non-vector R objects. """

    def __setattr__(self, name, value):
        if name == '_sexp':
            if not isinstance(value, rpy2.rinterface.Sexp):
                raise ValueError('_attr must contain an object that inherits from rpy2.rinterface.Sexp (not from %s)' % type(value))
        super(RObject, self).__setattr__(name, value)