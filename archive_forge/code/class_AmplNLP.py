from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
class AmplNLP(AslNLP):

    def __init__(self, nl_file, row_filename=None, col_filename=None):
        """
        AMPL nonlinear program interface.
        If row_filename and col_filename are not provided, the interface
        will see if files exist (with same name as nl_file but the .row
        and .col extensions)

        Parameters
        ----------
        nl_file: str
            filename of the NL-file containing the model
        row_filename: str, optional
            filename of .row file with identity of constraints
        col_filename: str, optional
            filename of .col file with identity of variables

        """
        super(AmplNLP, self).__init__(nl_file)
        if row_filename is None:
            tmp_filename = os.path.splitext(nl_file)[0] + '.row'
            if os.path.isfile(tmp_filename):
                row_filename = tmp_filename
        if col_filename is None:
            tmp_filename = os.path.splitext(nl_file)[0] + '.col'
            if os.path.isfile(tmp_filename):
                col_filename = tmp_filename
        self._rowfile = row_filename
        self._colfile = col_filename
        self._vidx_to_name = None
        self._name_to_vidx = None
        if col_filename is not None:
            self._vidx_to_name = self._build_component_names_list(col_filename)
            self._name_to_vidx = {self._vidx_to_name[vidx]: vidx for vidx in range(self._n_primals)}
        self._con_full_idx_to_name = None
        self._name_to_con_full_idx = None
        self._obj_name = None
        if row_filename is not None:
            all_names = self._build_component_names_list(row_filename)
            self._obj_name = all_names[-1]
            del all_names[-1]
            self._con_full_idx_to_name = all_names
            self._con_eq_idx_to_name = [all_names[self._con_eq_full_map[i]] for i in range(self._n_con_eq)]
            self._con_ineq_idx_to_name = [all_names[self._con_ineq_full_map[i]] for i in range(self._n_con_ineq)]
            self._name_to_con_full_idx = {all_names[cidx]: cidx for cidx in range(self._n_con_full)}
            self._name_to_con_eq_idx = {name: idx for idx, name in enumerate(self._con_eq_idx_to_name)}
            self._name_to_con_ineq_idx = {name: idx for idx, name in enumerate(self._con_ineq_idx_to_name)}

    def primals_names(self):
        """Returns ordered list with names of primal variables"""
        return list(self._vidx_to_name)

    @deprecated(msg='This method has been replaced with primals_names', version='6.0.0', remove_in='6.0')
    def variable_names(self):
        """Returns ordered list with names of primal variables"""
        return self.primals_names()

    def constraint_names(self):
        """Returns an ordered list with the names of all the constraints
        (corresponding to evaluate_constraints)"""
        return list(self._con_full_idx_to_name)

    def eq_constraint_names(self):
        """Returns ordered list with names of equality constraints only
        (corresponding to evaluate_eq_constraints)"""
        return list(self._con_eq_idx_to_name)

    def ineq_constraint_names(self):
        """Returns ordered list with names of inequality constraints only
        (corresponding to evaluate_ineq_constraints)"""
        return list(self._con_ineq_idx_to_name)

    @deprecated(msg='This method has been replaced with primal_idx', version='6.0.0', remove_in='6.0')
    def variable_idx(self, var_name):
        return self.primal_idx(var_name)

    def primal_idx(self, var_name):
        """
        Returns the index of the primal variable named var_name

        Parameters
        ----------
        var_name: str
            Name of primal variable

        Returns
        -------
        int

        """
        return self._name_to_vidx[var_name]

    def constraint_idx(self, con_name):
        """
        Returns the index of the constraint named con_name
        (corresponding to the order returned by evaluate_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int
        """
        return self._name_to_con_full_idx[con_name]

    def eq_constraint_idx(self, con_name):
        """
        Returns the index of the equality constraint named con_name
        (corresponding to the order returned by evaluate_eq_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int

        """
        return self._name_to_con_eq_idx[con_name]

    def ineq_constraint_idx(self, con_name):
        """
        Returns the index of the inequality constraint named con_name
        (corresponding to the order returned by evaluate_ineq_constraints)

        Parameters
        ----------
        con_name: str
            Name of constraint

        Returns
        -------
        int

        """
        return self._name_to_con_ineq_idx[con_name]

    @staticmethod
    def _build_component_names_list(filename):
        """Builds an ordered list of strings from a file
        containing strings on separate lines (e.g., the row
        and col files"""
        ordered_names = list()
        with open(filename, 'r') as f:
            for line in f:
                ordered_names.append(line.strip('\n'))
        return ordered_names