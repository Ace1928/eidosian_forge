import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Union
import sympy
import torch
from torch import SymInt
import torch.fx as fx
import torch.nn as nn
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols
from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
import torch.utils._pytree as pytree
import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess
from foo import FxModule
def _save_fx_default(current_name, folder_name, dump_example_input, gm, example_inputs):
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish difference graphs
    """
    from functorch.compile import aot_module_simplified

    def get_input_meta(args):
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            if type(arg) == int or type(arg) == float:
                input_meta.append((type(arg),))
            else:
                input_meta.append((type(arg), arg.shape, arg.stride(), arg.dtype, arg.device))
        return input_meta

    def graph_saver_helper(gm_to_save, args, type_name):
        global graph_index
        if len(gm_to_save.graph.nodes) == 0:
            log.log(logging.WARNING, 'No nodes in graph {%s}_{%s}_{%s}.', current_name, type_name, graph_index)
            return
        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())
        gm.recompile()
        input_meta = get_input_meta(args)
        isExist = os.path.exists(f'{folder_name}/{current_name}')
        if not isExist:
            os.makedirs(f'{folder_name}/{current_name}')
        gm.to_folder(f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}')
        pickle.dump(input_meta, open(f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input', 'wb'))
        if dump_example_input:
            torch.save(args, f'{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt')

    def graph_saver_forward(gm, fw_args):
        graph_saver_helper(gm, fw_args, 'forward')
        return gm

    def graph_saver_backward(gm, bw_args):
        graph_saver_helper(gm, bw_args, 'backward')
        global graph_index
        graph_index += 1
        return gm

    def graph_saver_joint(gm, joint_args):
        graph_saver_helper(gm, joint_args, 'joint')
        return default_partition(gm, joint_args)
    return aot_module_simplified(gm, example_inputs, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward, partition_fn=graph_saver_joint, decompositions=default_decompositions)