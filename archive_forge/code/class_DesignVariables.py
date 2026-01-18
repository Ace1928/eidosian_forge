import itertools
class DesignVariables(VariablesWithIndices):
    """
    Define design variables
    """

    def __init__(self):
        super().__init__()

    def set_variable_name_list(self, variable_name_list):
        """
        Specify variable names with its full name.

        Parameters
        ----------
        variable_name_list: a ``list`` of ``string``, containing the variable names with indices,
            for e.g. "C['CA', 23, 0]".
        """
        super().set_variable_name_list(variable_name_list)

    def add_variables(self, var_name, indices=None, time_index_position=None, values=None, lower_bounds=None, upper_bounds=None):
        """

        Parameters
        ----------
        var_name: a ``list`` of var names
        indices: a ``dict`` containing indices
            if default (None), no extra indices needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        values: a ``list`` containing values which has the same shape of flattened variables
            default choice is None, means there is no give nvalues
        lower_bounds: a ``list`` of lower bounds. If given a scalar number, it is set as the lower bounds for all variables.
        upper_bounds: a ``list`` of upper bounds. If given a scalar number, it is set as the upper bounds for all variables.
        """
        super().add_variables(var_name=var_name, indices=indices, time_index_position=time_index_position, values=values, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    def update_values(self, new_value_dict):
        """
        Update values of variables. Used for defining values for design variables of different experiments.

        Parameters
        ----------
        new_value_dict: a ``dict`` containing the new values for the variables.
            for e.g., {"C['CA', 23, 0]": 0.5, "C['CA', 24, 0]": 0.6}
        """
        for key in new_value_dict:
            if key not in self.variable_names:
                raise ValueError('Variable not in the set: ', key)
            self.variable_names_value[key] = new_value_dict[key]