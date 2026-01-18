import torch
def _transpose_with_info(values, _transpose_info):
    row_indices_t, row_offsets_t, column_indices_t, perm = _transpose_info
    values_t = values[:, perm]
    return (row_indices_t, values_t, row_offsets_t, column_indices_t)