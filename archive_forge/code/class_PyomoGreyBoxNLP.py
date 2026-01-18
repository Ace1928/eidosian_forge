import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
class PyomoGreyBoxNLP(NLP):

    def __init__(self, pyomo_model):
        greybox_components = []
        try:
            for greybox in pyomo_model.component_objects(ExternalGreyBoxBlock, descend_into=True):
                greybox.parent_block().reclassify_component_type(greybox, pyo.Block)
                greybox_components.append(greybox)
            self._pyomo_model = pyomo_model
            self._pyomo_nlp = PyomoNLP(pyomo_model)
        finally:
            for greybox in greybox_components:
                greybox.parent_block().reclassify_component_type(greybox, ExternalGreyBoxBlock)
        greybox_data = []
        for greybox in greybox_components:
            greybox_data.extend((data for data in greybox.values() if data.active))
        if len(greybox_data) > 1:
            raise NotImplementedError('The PyomoGreyBoxModel interface has not been tested with Pyomo models that contain more than one ExternalGreyBoxBlock. Currently, only a single block is supported.')
        if self._pyomo_nlp.n_primals() == 0:
            raise ValueError('No variables were found in the Pyomo part of the model. PyomoGreyBoxModel requires at least one variable to be active in a Pyomo objective or constraint')
        "\n        for data in greybox_data:\n            c = data._ex_model.model_capabilities()\n            if (c.n_equality_constraints() > 0                and not c.supports_jacobian_equality_constraints)                or (c.n_equality_constraints() > 0                and not c.supports_jacobian_equality_constraints)\n                raise NotImplementedError('PyomoGreyBoxNLP does not support models'\n                                          ' without explicit Jacobian support')\n        "
        self._n_greybox_primals = 0
        self._greybox_primals_names = []
        self._greybox_constraints_names = []
        n_primals = self._pyomo_nlp.n_primals()
        greybox_primals = []
        self._vardata_to_idx = ComponentMap(self._pyomo_nlp._vardata_to_idx)
        for data in greybox_data:
            for v in data.inputs.values():
                if v.fixed:
                    raise NotImplementedError('Found a grey box model input that is fixed: {}. This interface does not currently support fixed variables. Please add a constraint instead.'.format(v.getname(fully_qualified=True)))
            for v in data.outputs.values():
                if v.fixed:
                    raise NotImplementedError('Found a grey box model output that is fixed: {}. This interface does not currently support fixed variables. Please add a constraint instead.'.format(v.getname(fully_qualified=True)))
            block_name = data.getname()
            for nm in data._ex_model.equality_constraint_names():
                self._greybox_constraints_names.append('{}.{}'.format(block_name, nm))
            for nm in data._ex_model.output_names():
                self._greybox_constraints_names.append('{}.{}_con'.format(block_name, nm))
            for var in data.component_data_objects(pyo.Var):
                if var not in self._vardata_to_idx:
                    self._vardata_to_idx[var] = n_primals
                    n_primals += 1
                    greybox_primals.append(var)
                    self._greybox_primals_names.append(var.getname(fully_qualified=True))
        self._n_greybox_primals = len(greybox_primals)
        self._greybox_primal_variables = greybox_primals
        self._greybox_primals_lb = np.zeros(self._n_greybox_primals)
        self._greybox_primals_ub = np.zeros(self._n_greybox_primals)
        self._init_greybox_primals = np.zeros(self._n_greybox_primals)
        for i, var in enumerate(greybox_primals):
            if var.value is not None:
                self._init_greybox_primals[i] = var.value
            self._greybox_primals_lb[i] = -np.inf if var.lb is None else var.lb
            self._greybox_primals_ub[i] = np.inf if var.ub is None else var.ub
        self._greybox_primals_lb.flags.writeable = False
        self._greybox_primals_ub.flags.writeable = False
        self._greybox_primals = self._init_greybox_primals.copy()
        self._cached_greybox_constraints = None
        self._cached_greybox_jac = None
        con_offset = self._pyomo_nlp.n_constraints()
        self._external_greybox_helpers = []
        for data in greybox_data:
            h = _ExternalGreyBoxModelHelper(data, self._vardata_to_idx, con_offset)
            self._external_greybox_helpers.append(h)
            con_offset += h.n_residuals()
        self._n_greybox_constraints = con_offset - self._pyomo_nlp.n_constraints()
        assert len(self._greybox_constraints_names) == self._n_greybox_constraints
        self.set_primals(self.get_primals())
        need_scaling = False
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        if self._obj_scaling is None:
            self._obj_scaling = 1.0
        else:
            need_scaling = True
        self._primals_scaling = np.ones(self.n_primals())
        scaling_suffix = self._pyomo_nlp._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            need_scaling = True
            for i, v in enumerate(self.get_pyomo_variables()):
                if v in scaling_suffix:
                    self._primals_scaling[i] = scaling_suffix[v]
        self._constraints_scaling = []
        pyomo_nlp_scaling = self._pyomo_nlp.get_constraints_scaling()
        if pyomo_nlp_scaling is None:
            pyomo_nlp_scaling = np.ones(self._pyomo_nlp.n_constraints())
        else:
            need_scaling = True
        self._constraints_scaling.append(pyomo_nlp_scaling)
        for h in self._external_greybox_helpers:
            tmp_scaling = h.get_residual_scaling()
            if tmp_scaling is None:
                tmp_scaling = np.ones(h.n_residuals())
            else:
                need_scaling = True
            self._constraints_scaling.append(tmp_scaling)
        if need_scaling:
            self._constraints_scaling = np.concatenate(self._constraints_scaling)
        else:
            self._obj_scaling = None
            self._primals_scaling = None
            self._constraints_scaling = None
        self._init_greybox_duals = np.zeros(self._n_greybox_constraints)
        self._init_greybox_primals.flags.writeable = False
        self._init_greybox_duals.flags.writeable = False
        self._greybox_duals = self._init_greybox_duals.copy()
        self._evaluate_greybox_jacobians_and_cache_if_necessary()
        self._nnz_greybox_jac = len(self._cached_greybox_jac.data)
        self.set_duals(self.get_duals())
        try:
            self._evaluate_greybox_hessians_and_cache_if_necessary()
            self._nnz_greybox_hess = len(self._cached_greybox_hess.data)
        except (AttributeError, NotImplementedError):
            self._nnz_greybox_hess = None

    def _invalidate_greybox_primals_cache(self):
        self._greybox_constraints_cached = False
        self._greybox_jac_cached = False
        self._greybox_hess_cached = False

    def _invalidate_greybox_duals_cache(self):
        self._greybox_hess_cached = False

    def n_primals(self):
        return self._pyomo_nlp.n_primals() + self._n_greybox_primals

    def n_constraints(self):
        return self._pyomo_nlp.n_constraints() + self._n_greybox_constraints

    def n_eq_constraints(self):
        return self._pyomo_nlp.n_eq_constraints() + self._n_greybox_constraints

    def n_ineq_constraints(self):
        return self._pyomo_nlp.n_ineq_constraints()

    def nnz_jacobian(self):
        return self._pyomo_nlp.nnz_jacobian() + self._nnz_greybox_jac

    def nnz_jacobian_eq(self):
        return self._pyomo_nlp.nnz_jacobian_eq() + self._nnz_greybox_jac

    def nnz_hessian_lag(self):
        return self._pyomo_nlp.nnz_hessian_lag() + self._nnz_greybox_hess

    def primals_lb(self):
        return np.concatenate((self._pyomo_nlp.primals_lb(), self._greybox_primals_lb))

    def primals_ub(self):
        return np.concatenate((self._pyomo_nlp.primals_ub(), self._greybox_primals_ub))

    def constraints_lb(self):
        return np.concatenate((self._pyomo_nlp.constraints_lb(), np.zeros(self._n_greybox_constraints, dtype=np.float64)))

    def constraints_ub(self):
        return np.concatenate((self._pyomo_nlp.constraints_ub(), np.zeros(self._n_greybox_constraints, dtype=np.float64)))

    def init_primals(self):
        return np.concatenate((self._pyomo_nlp.init_primals(), self._init_greybox_primals))

    def init_duals(self):
        return np.concatenate((self._pyomo_nlp.init_duals(), self._init_greybox_duals))

    def init_duals_eq(self):
        return np.concatenate((self._pyomo_nlp.init_duals_eq(), self._init_greybox_duals))

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)
        elif vector_type == 'eq_constraints' or vector_type == 'duals_eq':
            return np.zeros(self.n_eq_constraints(), dtype=np.float64)
        elif vector_type == 'ineq_constraints' or vector_type == 'duals_ineq':
            return np.zeros(self.n_ineq_constraints(), dtype=np.float64)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    def set_primals(self, primals):
        self._invalidate_greybox_primals_cache()
        self._pyomo_nlp.set_primals(primals[:self._pyomo_nlp.n_primals()])
        np.copyto(self._greybox_primals, primals[self._pyomo_nlp.n_primals():])
        for external in self._external_greybox_helpers:
            external.set_primals(primals)

    def get_primals(self):
        return np.concatenate((self._pyomo_nlp.get_primals(), self._greybox_primals))

    def set_duals(self, duals):
        self._invalidate_greybox_duals_cache()
        self._pyomo_nlp.set_duals(duals[:self._pyomo_nlp.n_constraints()])
        np.copyto(self._greybox_duals, duals[self._pyomo_nlp.n_constraints():])
        for h in self._external_greybox_helpers:
            h.set_duals(duals)

    def get_duals(self):
        return np.concatenate((self._pyomo_nlp.get_duals(), self._greybox_duals))

    def set_duals_eq(self, duals):
        raise NotImplementedError('set_duals_eq not implemented for PyomoGreyBoxNLP')
        '\n        #self._invalidate_greybox_duals_cache()\n\n        # set the duals for the pyomo part of the nlp\n        self._pyomo_nlp.set_duals_eq(\n            duals[:self._pyomo_nlp.n_equality_constraints()])\n\n        # set the duals for the greybox part of the nlp\n        np.copyto(self._greybox_duals, duals[self._pyomo_nlp.n_equality_constraints():])\n        # set the duals in the helpers for the hessian computation\n        for h in self._external_greybox_helpers:\n            h.set_duals_eq(duals)\n        '

    def get_duals_eq(self):
        raise NotImplementedError('get_duals_eq not implemented for PyomoGreyBoxNLP')
        '\n        # return the duals for the pyomo part of the nlp\n        # concatenated with the greybox part\n        return np.concatenate((\n            self._pyomo_nlp.get_duals_eq(),\n            self._greybox_duals,\n        ))\n        '

    def set_obj_factor(self, obj_factor):
        self._pyomo_nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._pyomo_nlp.get_obj_factor()

    def get_obj_scaling(self):
        return self._obj_scaling

    def get_primals_scaling(self):
        return self._primals_scaling

    def get_constraints_scaling(self):
        return self._constraints_scaling

    def evaluate_objective(self):
        return self._pyomo_nlp.evaluate_objective()

    def evaluate_grad_objective(self, out=None):
        return np.concatenate((self._pyomo_nlp.evaluate_grad_objective(out), np.zeros(self._n_greybox_primals)))

    def _evaluate_greybox_constraints_and_cache_if_necessary(self):
        if self._greybox_constraints_cached:
            return
        self._cached_greybox_constraints = np.concatenate(tuple((external.evaluate_residuals() for external in self._external_greybox_helpers)))
        self._greybox_constraints_cached = True

    def evaluate_constraints(self, out=None):
        self._evaluate_greybox_constraints_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self.n_constraints():
                raise RuntimeError('Called evaluate_constraints with an invalid "out" argument - should take an ndarray of size {}'.format(self.n_constraints()))
            self._pyomo_nlp.evaluate_constraints(out[:-self._n_greybox_constraints])
            np.copyto(out[-self._n_greybox_constraints:], self._cached_greybox_constraints)
            return out
        else:
            return np.concatenate((self._pyomo_nlp.evaluate_constraints(), self._cached_greybox_constraints))

    def evaluate_eq_constraints(self, out=None):
        raise NotImplementedError('Not yet implemented for PyomoGreyBoxNLP')
        '\n        self._evaluate_greybox_constraints_and_cache_if_necessary()\n\n        if out is not None:\n            if not isinstance(out, np.ndarray)                or out.size != self.n_eq_constraints():\n                raise RuntimeError(\n                    \'Called evaluate_eq_constraints with an invalid\'\n                    \' "out" argument - should take an ndarray of \'\n                    \'size {}\'.format(self.n_eq_constraints()))\n            self._pyomo_nlp.evaluate_eq_constraints(\n                out[:-self._n_greybox_constraints])\n            np.copyto(out[-self._n_greybox_constraints:], self._cached_greybox_constraints)\n            return out\n        else:\n            return np.concatenate((\n                self._pyomo_nlp.evaluate_eq_constraints(),\n                self._cached_greybox_constraints,\n            ))\n        '

    def _evaluate_greybox_jacobians_and_cache_if_necessary(self):
        if self._greybox_jac_cached:
            return
        jac = BlockMatrix(len(self._external_greybox_helpers), 1)
        for i, external in enumerate(self._external_greybox_helpers):
            jac.set_block(i, 0, external.evaluate_jacobian())
        self._cached_greybox_jac = jac.tocoo()
        self._greybox_jac_cached = True

    def evaluate_jacobian(self, out=None):
        self._evaluate_greybox_jacobians_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self.n_constraints() or out.shape[1] != self.n_primals() or (out.nnz != self.nnz_jacobian()):
                raise RuntimeError('evaluate_jacobian called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self.n_constraints(), self.n_primals(), self.nnz_jacobian()))
            n_pyomo_constraints = self.n_constraints() - self._n_greybox_constraints
            self._pyomo_nlp.evaluate_jacobian(out=coo_matrix((out.data[:-self._nnz_greybox_jac], (out.row[:-self._nnz_greybox_jac], out.col[:-self._nnz_greybox_jac])), shape=(n_pyomo_constraints, self._pyomo_nlp.n_primals())))
            np.copyto(out.data[-self._nnz_greybox_jac:], self._cached_greybox_jac.data)
            return out
        else:
            base = self._pyomo_nlp.evaluate_jacobian()
            base = coo_matrix((base.data, (base.row, base.col)), shape=(base.shape[0], self.n_primals()))
            jac = BlockMatrix(2, 1)
            jac.set_block(0, 0, base)
            jac.set_block(1, 0, self._cached_greybox_jac)
            return jac.tocoo()
    '\n    def evaluate_jacobian_eq(self, out=None):\n        raise NotImplementedError()\n        self._evaluate_greybox_jacobians_and_cache_if_necessary()\n\n        if out is not None:\n            if ( not isinstance(out, coo_matrix)\n                 or out.shape[0] != self.n_eq_constraints()\n                 or out.shape[1] != self.n_primals()\n                 or out.nnz != self.nnz_jacobian_eq() ):\n                raise RuntimeError(\n                    \'evaluate_jacobian called with an "out" argument\'\n                    \' that is invalid. This should be a coo_matrix with\'\n                    \' shape=({},{}) and nnz={}\'\n                    .format(self.n_eq_constraints(), self.n_primals(),\n                            self.nnz_jacobian_eq()))\n            self._pyomo_nlp.evaluate_jacobian_eq(\n                coo_matrix((out.data[:-self._nnz_greybox_jac],\n                            (out.row[:-self._nnz_greybox_jac],\n                             out.col[:-self._nnz_greybox_jac])))\n            )\n            np.copyto(out.data[-self._nnz_greybox_jac],\n                      self._cached_greybox_jac.data)\n            return out\n        else:\n            base = self._pyomo_nlp.evaluate_jacobian_eq()\n            # TODO: Doesn\'t this need a "shape" specification?\n            return coo_matrix((\n                np.concatenate((base.data, self._cached_greybox_jac.data)),\n                ( np.concatenate((base.row, self._cached_greybox_jac.row)),\n                  np.concatenate((base.col, self._cached_greybox_jac.col)) )\n            ))\n    '

    def _evaluate_greybox_hessians_and_cache_if_necessary(self):
        if self._greybox_hess_cached:
            return
        data = list()
        irow = list()
        jcol = list()
        for external in self._external_greybox_helpers:
            hess = external.evaluate_hessian()
            data.append(hess.data)
            irow.append(hess.row)
            jcol.append(hess.col)
        data = np.concatenate(data)
        irow = np.concatenate(irow)
        jcol = np.concatenate(jcol)
        self._cached_greybox_hess = coo_matrix((data, (irow, jcol)), shape=(self.n_primals(), self.n_primals()))
        self._greybox_hess_cached = True

    def evaluate_hessian_lag(self, out=None):
        self._evaluate_greybox_hessians_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self.n_primals() or out.shape[1] != self.n_primals() or (out.nnz != self.nnz_hessian_lag()):
                raise RuntimeError('evaluate_hessian_lag called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self.n_primals(), self.n_primals(), self.nnz_hessian()))
            self._pyomo_nlp.evaluate_hessian_lag(out=coo_matrix((out.data[:-self._nnz_greybox_hess], (out.row[:-self._nnz_greybox_hess], out.col[:-self._nnz_greybox_hess])), shape=(self._pyomo_nlp.n_primals(), self._pyomo_nlp.n_primals())))
            np.copyto(out.data[-self._nnz_greybox_hess:], self._cached_greybox_hess.data)
            return out
        else:
            hess = self._pyomo_nlp.evaluate_hessian_lag()
            data = np.concatenate((hess.data, self._cached_greybox_hess.data))
            row = np.concatenate((hess.row, self._cached_greybox_hess.row))
            col = np.concatenate((hess.col, self._cached_greybox_hess.col))
            hess = coo_matrix((data, (row, col)), shape=(self.n_primals(), self.n_primals()))
            return hess

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('Todo: implement this')

    @deprecated(msg='This method has been replaced with primals_names', version='6.0.0', remove_in='6.0')
    def variable_names(self):
        return self.primals_names()

    def primals_names(self):
        names = list(self._pyomo_nlp.variable_names())
        names.extend(self._greybox_primals_names)
        return names

    def constraint_names(self):
        names = list(self._pyomo_nlp.constraint_names())
        names.extend(self._greybox_constraints_names)
        return names

    def pyomo_model(self):
        """
        Return optimization model
        """
        return self._pyomo_model

    def get_pyomo_objective(self):
        """
        Return an instance of the active objective function on the Pyomo model.
        (there can be only one)
        """
        return self._pyomo_nlp.get_pyomo_objective()

    def get_pyomo_variables(self):
        """
        Return an ordered list of the Pyomo VarData objects in
        the order corresponding to the primals
        """
        return self._pyomo_nlp.get_pyomo_variables() + self._greybox_primal_variables

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        raise NotImplementedError('returning list of all constraints when using an external model is TBD')

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
        if 'dual' in model_suffixes:
            model_suffixes['dual'].clear()
        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(zip(variables, bound_multipliers[0]))
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(zip(variables, bound_multipliers[1]))