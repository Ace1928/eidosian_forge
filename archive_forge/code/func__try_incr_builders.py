import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action):
    if isinstance(formula_like, DesignInfo):
        return (design_matrix_builders([[]], data_iter_maker, eval_env, NA_action)[0], formula_like)
    if isinstance(formula_like, tuple) and len(formula_like) == 2 and isinstance(formula_like[0], DesignInfo) and isinstance(formula_like[1], DesignInfo):
        return formula_like
    if hasattr(formula_like, '__patsy_get_model_desc__'):
        formula_like = formula_like.__patsy_get_model_desc__(eval_env)
        if not isinstance(formula_like, ModelDesc):
            raise PatsyError('bad value from %r.__patsy_get_model_desc__' % (formula_like,))
    if not six.PY3 and isinstance(formula_like, unicode):
        try:
            formula_like = formula_like.encode('ascii')
        except UnicodeEncodeError:
            raise PatsyError("On Python 2, formula strings must be either 'str' objects, or else 'unicode' objects containing only ascii characters. You passed a unicode string with non-ascii characters. I'm afraid you'll have to either switch to ascii-only, or else upgrade to Python 3.")
    if isinstance(formula_like, str):
        formula_like = ModelDesc.from_formula(formula_like)
    if isinstance(formula_like, ModelDesc):
        assert isinstance(eval_env, EvalEnvironment)
        return design_matrix_builders([formula_like.lhs_termlist, formula_like.rhs_termlist], data_iter_maker, eval_env, NA_action)
    else:
        return None