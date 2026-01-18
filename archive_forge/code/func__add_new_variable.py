from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
def _add_new_variable(initial_value, var_name_v2, var_name_v1, var_map, var_names_map):
    """Creates a new variable and add it to the variable maps."""
    var = tf.Variable(initial_value, name=var_name_v2)
    var_map[var_name_v2] = var
    var_names_map[var_name_v2] = var_name_v1