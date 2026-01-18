from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
def _convert_variables_in_ckpt(opt_name_v1, reader, variable_names, var_map, var_names_map, est_type):
    """Converts all variables in checkpoint from v1 to v2."""
    global_step = None
    hp_ckpt = None
    if opt_name_v1 == 'Adam':
        global_step = reader.get_tensor('global_step')
    if opt_name_v1 in HP_IN_CKPT:
        hp_ckpt = HP_IN_CKPT[opt_name_v1]
    opt_name_v2 = OPT_NAME_V1_TO_V2[opt_name_v1]
    for var_name in variable_names:
        if hp_ckpt and any((hp_name in var_name for hp_name in hp_ckpt)):
            for hp_name in hp_ckpt:
                if hp_name in var_name:
                    var_name_v2 = hp_ckpt[hp_name]
                    tensor = reader.get_tensor(var_name)
                    tensor = tf.math.pow(tensor, 1.0 / global_step)
                    _add_new_variable(tensor, var_name_v2, var_name, var_map, var_names_map)
                    break
        elif opt_name_v1 in var_name:
            suffix_mapping = OPT_VAR_NAME_V1_TO_V2[opt_name_v1]
            suffix_v1 = var_name.rsplit('/')[-1]
            suffix_v2 = suffix_mapping[suffix_v1]
            if suffix_v2:
                if est_type == 'dnn':
                    idx = var_name.rfind('t_0')
                    _add_opt_variable(opt_name_v2, var_name, idx, suffix_v2, reader, var_map, var_names_map)
                elif est_type == 'linear':
                    idx = var_name.rfind('part_0')
                    _add_opt_variable(opt_name_v2, var_name, idx, suffix_v2, reader, var_map, var_names_map)
                else:
                    idx = var_name.rfind(suffix_v1)
                    _add_opt_variable(opt_name_v2, var_name, idx, suffix_v2, reader, var_map, var_names_map)
        else:
            tensor = reader.get_tensor(var_name)
            _add_new_variable(tensor, var_name, var_name, var_map, var_names_map)