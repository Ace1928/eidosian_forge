import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _generate_op_module_signature(root_namespace, module_name, op_code_gen_func):
    """
    Generate op functions created by `op_code_gen_func` and write to the source file
    of `root_namespace.module_name.[submodule_name]`,
    where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.

    Parameters
    ----------
    root_namespace : str
        Top level module name, `mxnet` in the current cases.
    module_name : str
        Second level module name, `ndarray` and `symbol` in the current cases.
    op_code_gen_func : function
        Function for creating op functions for `ndarray` and `symbol` modules.
    """
    license_lines = ['# Licensed to the Apache Software Foundation (ASF) under one', '# or more contributor license agreements.  See the NOTICE file', '# distributed with this work for additional information', '# regarding copyright ownership.  The ASF licenses this file', '# to you under the Apache License, Version 2.0 (the', '# "License"); you may not use this file except in compliance', '# with the License.  You may obtain a copy of the License at', '#', '#   http://www.apache.org/licenses/LICENSE-2.0', '#', '# Unless required by applicable law or agreed to in writing,', '# software distributed under the License is distributed on an', '# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY', '# KIND, either express or implied.  See the License for the', '# specific language governing permissions and limitations', '# under the License.', '']
    license_str = os.linesep.join(license_lines)

    def get_module_file(module_name):
        """Return the generated module file based on module name."""
        path = os.path.dirname(__file__)
        module_path = module_name.split('.')
        module_path[-1] = 'gen_' + module_path[-1]
        file_name = os.path.join(path, '..', *module_path) + '.py'
        module_file = open(file_name, 'w', encoding='utf-8')
        dependencies = {'symbol': ['from ._internal import SymbolBase', 'from ..base import _Null'], 'ndarray': ['from ._internal import NDArrayBase', 'from ..base import _Null']}
        module_file.write('# coding: utf-8')
        module_file.write(license_str)
        module_file.write('# File content is auto-generated. Do not modify.' + os.linesep)
        module_file.write('# pylint: skip-file' + os.linesep)
        module_file.write(os.linesep.join(dependencies[module_name.split('.')[1]]))
        return module_file

    def write_all_str(module_file, module_all_list):
        """Write the proper __all__ based on available operators."""
        module_file.write(os.linesep)
        module_file.write(os.linesep)
        all_str = '__all__ = [' + ', '.join(["'%s'" % s for s in module_all_list]) + ']'
        module_file.write(all_str)
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_name = py_str(plist[i])
        if not _is_np_op(op_name):
            op_names.append(op_name)
    module_op_file = get_module_file('%s.%s.op' % (root_namespace, module_name))
    module_op_all = []
    module_internal_file = get_module_file('%s.%s._internal' % (root_namespace, module_name))
    module_internal_all = []
    submodule_dict = {}
    for op_name_prefix in _OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] = (get_module_file('%s.%s.%s' % (root_namespace, module_name, op_name_prefix[1:-1])), [])
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        op_name_prefix = _get_op_name_prefix(name)
        if len(op_name_prefix) > 0:
            func_name = name[len(op_name_prefix):]
            cur_module_file, cur_module_all = submodule_dict[op_name_prefix]
        elif name.startswith('_'):
            func_name = name
            cur_module_file = module_internal_file
            cur_module_all = module_internal_all
        else:
            func_name = name
            cur_module_file = module_op_file
            cur_module_all = module_op_all
        code, _ = op_code_gen_func(hdl, name, func_name, True)
        cur_module_file.write(os.linesep)
        cur_module_file.write(code)
        cur_module_all.append(func_name)
    for submodule_f, submodule_all in submodule_dict.values():
        write_all_str(submodule_f, submodule_all)
        submodule_f.close()
    write_all_str(module_op_file, module_op_all)
    module_op_file.close()
    write_all_str(module_internal_file, module_internal_all)
    module_internal_file.close()