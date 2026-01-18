import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def _show_defined_functions(saved_model_dir, meta_graphs):
    """Prints the callable concrete and polymorphic functions of the Saved Model.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    meta_graphs: Already-extracted MetaGraphDef of the SavedModel.
  """
    has_object_graph_def = False
    for meta_graph_def in meta_graphs:
        has_object_graph_def |= meta_graph_def.HasField('object_graph_def')
    if not has_object_graph_def:
        return
    with ops_lib.Graph().as_default():
        trackable_object = load.load(saved_model_dir)
    print('\nConcrete Functions:', end='')
    children = list(save._AugmentedGraphView(trackable_object).list_children(trackable_object))
    children = sorted(children, key=lambda x: x.name)
    for name, child in children:
        concrete_functions = []
        if isinstance(child, defun.ConcreteFunction):
            concrete_functions.append(child)
        elif isinstance(child, def_function.Function):
            concrete_functions.extend(child._list_all_concrete_functions_for_serialization())
        else:
            continue
        print("\n  Function Name: '%s'" % name)
        concrete_functions = sorted(concrete_functions, key=lambda x: x.name)
        for index, concrete_function in enumerate(concrete_functions, 1):
            args, kwargs = (None, None)
            if concrete_function.structured_input_signature:
                args, kwargs = concrete_function.structured_input_signature
            elif concrete_function._arg_keywords:
                args = concrete_function._arg_keywords
            if args:
                print('    Option #%d' % index)
                print('      Callable with:')
                _print_args(args, indent=4)
            if kwargs:
                _print_args(kwargs, 'Named Argument', indent=4)