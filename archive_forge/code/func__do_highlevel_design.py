import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type):
    if return_type == 'dataframe' and (not have_pandas):
        raise PatsyError('pandas.DataFrame was requested, but pandas is not installed')
    if return_type not in ('matrix', 'dataframe'):
        raise PatsyError("unrecognized output type %r, should be 'matrix' or 'dataframe'" % (return_type,))

    def data_iter_maker():
        return iter([data])
    design_infos = _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
    if design_infos is not None:
        return build_design_matrices(design_infos, data, NA_action=NA_action, return_type=return_type)
    else:
        if isinstance(formula_like, tuple):
            if len(formula_like) != 2:
                raise PatsyError("don't know what to do with a length %s matrices tuple" % (len(formula_like),))
            lhs, rhs = formula_like
        else:
            lhs, rhs = (None, asarray_or_pandas(formula_like, subok=True))

        def _regularize_matrix(m, default_column_prefix):
            di = DesignInfo.from_array(m, default_column_prefix)
            if have_pandas and isinstance(m, (pandas.Series, pandas.DataFrame)):
                orig_index = m.index
            else:
                orig_index = None
            if return_type == 'dataframe':
                m = atleast_2d_column_default(m, preserve_pandas=True)
                m = pandas.DataFrame(m)
                m.columns = di.column_names
                m.design_info = di
                return (m, orig_index)
            else:
                return (DesignMatrix(m, di), orig_index)
        rhs, rhs_orig_index = _regularize_matrix(rhs, 'x')
        if lhs is None:
            lhs = np.zeros((rhs.shape[0], 0), dtype=float)
        lhs, lhs_orig_index = _regularize_matrix(lhs, 'y')
        assert isinstance(getattr(lhs, 'design_info', None), DesignInfo)
        assert isinstance(getattr(rhs, 'design_info', None), DesignInfo)
        if lhs.shape[0] != rhs.shape[0]:
            raise PatsyError('shape mismatch: outcome matrix has %s rows, predictor matrix has %s rows' % (lhs.shape[0], rhs.shape[0]))
        if rhs_orig_index is not None and lhs_orig_index is not None:
            if not rhs_orig_index.equals(lhs_orig_index):
                raise PatsyError('index mismatch: outcome and predictor have incompatible indexes')
        if return_type == 'dataframe':
            if rhs_orig_index is not None and lhs_orig_index is None:
                lhs.index = rhs.index
            if rhs_orig_index is None and lhs_orig_index is not None:
                rhs.index = lhs.index
        return (lhs, rhs)