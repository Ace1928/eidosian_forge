import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
@staticmethod
def _extract_symbol(line) -> Optional[Symbol]:
    entries = line.split('@')
    ret_str = entries[0]
    func_str = entries[1]
    ret_strs = ret_str.split()
    if ret_strs[1] == 'internal':
        return None
    ret_type = convert_type(ret_strs[1])
    if ret_type is None:
        return None
    func_strs = func_str.split('(')
    func_name = func_strs[0].replace('@', '')
    op_name = func_name.replace('__nv_', '')
    if 'ieee' in op_name:
        return None
    arg_strs = func_strs[1].split(',')
    arg_types = []
    arg_names = []
    for i, arg_str in enumerate(arg_strs):
        arg_type = convert_type(arg_str.split()[0])
        if arg_type is None:
            return None
        arg_name = 'arg' + str(i)
        arg_types.append(arg_type)
        arg_names.append(arg_name)
    if op_name == 'sad':
        arg_types[-1] = to_unsigned(arg_types[-1])
    elif op_name.startswith('u'):
        ret_type = to_unsigned(ret_type)
        for i, arg_type in enumerate(arg_types):
            arg_types[i] = to_unsigned(arg_type)
    return Symbol(func_name, op_name, ret_type, arg_names, arg_types)