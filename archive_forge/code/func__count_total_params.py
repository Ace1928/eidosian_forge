import argparse
import re
import sys
from absl import app
import numpy as np
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import flags
from tensorflow.python.training import py_checkpoint_reader
def _count_total_params(reader, count_exclude_pattern=''):
    """Count total number of variables."""
    var_to_shape_map = reader.get_variable_to_shape_map()
    if count_exclude_pattern:
        regex_pattern = re.compile(count_exclude_pattern)
        new_var_to_shape_map = {}
        exclude_num_tensors = 0
        exclude_num_params = 0
        for v in var_to_shape_map:
            if regex_pattern.search(v):
                exclude_num_tensors += 1
                exclude_num_params += np.prod(var_to_shape_map[v])
            else:
                new_var_to_shape_map[v] = var_to_shape_map[v]
        var_to_shape_map = new_var_to_shape_map
        print('# Excluding %d tensors (%d params) that match %s when counting.' % (exclude_num_tensors, exclude_num_params, count_exclude_pattern))
    var_sizes = [np.prod(var_to_shape_map[v]) for v in var_to_shape_map]
    return np.sum(var_sizes, dtype=int)