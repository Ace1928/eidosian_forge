from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def __DELETE_OP_DEFAULT(key_set, command, exist_conf):
    new_conf = exist_conf
    trival_cmd_key_set, dict_list_cmd_key_set = get_key_sets(command)
    if len(key_set) == len(trival_cmd_key_set) and len(dict_list_cmd_key_set) == 0:
        new_conf = []
        return (True, new_conf)
    trival_cmd_key_not_key_set = trival_cmd_key_set.difference(key_set)
    for key in trival_cmd_key_not_key_set:
        command_val = command.get(key, None)
        new_conf_val = new_conf.get(key, None)
        if command_val == new_conf_val:
            new_conf.pop(key, None)
    return (False, new_conf)