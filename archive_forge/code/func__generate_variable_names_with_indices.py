import itertools
def _generate_variable_names_with_indices(self, var_name, indices=None, time_index_position=None):
    """
        Used for generating string names with indices.

        Parameters
        ----------
        var_name: a ``list`` of var names
        indices: a ``dict`` containing indices
            if default (None), no extra indices needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        """
    all_index_list = []
    if indices:
        for index_pointer in indices:
            all_index_list.append(indices[index_pointer])
    all_variable_indices = list(itertools.product(*all_index_list))
    added_names = []
    for index_instance in all_variable_indices:
        var_name_index_string = var_name + '['
        for i, idx in enumerate(index_instance):
            var_name_index_string += str(idx)
            if i == len(index_instance) - 1:
                var_name_index_string += ']'
            else:
                var_name_index_string += ','
        self.variable_names.append(var_name_index_string)
        added_names.append(var_name_index_string)
    return added_names