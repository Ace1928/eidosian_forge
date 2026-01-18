from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
@register_model(types.PolynomialType)
class PolynomialModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('coef', fe_type.coef), ('domain', fe_type.domain), ('window', fe_type.window)]
        super(PolynomialModel, self).__init__(dmm, fe_type, members)