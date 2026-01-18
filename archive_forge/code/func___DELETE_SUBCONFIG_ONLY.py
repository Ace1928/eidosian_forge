from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def __DELETE_SUBCONFIG_ONLY(key_set, command, exist_conf):
    new_conf = exist_conf
    nu, dict_list_cmd_key_set = get_key_sets(command)
    nu, dict_list_exist_key_set = get_key_sets(exist_conf)
    common_dict_list_key_set = dict_list_cmd_key_set.intersection(dict_list_exist_key_set)
    for key in common_dict_list_key_set:
        new_conf.pop(key, None)
    return (True, new_conf)