import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _fill_meta_graph_def(meta_graph_def: meta_graph_pb2.MetaGraphDef, saveable_view: _SaveableView, signature_functions: Dict[str, Callable[..., Any]], namespace_whitelist: List[str], save_custom_gradients: bool, defaults=None) -> Tuple[_AssetInfo, ops.Graph]:
    """Generates a MetaGraph which calls `signature_functions`.

  Args:
    meta_graph_def: The MetaGraphDef proto to fill.
    saveable_view: The _SaveableView being exported.
    signature_functions: A dictionary mapping signature keys to concrete
      functions containing signatures to add to the MetaGraph.
    namespace_whitelist: List of strings containing whitelisted op namespaces.
    save_custom_gradients: Whether to save custom gradients.
    defaults: A dictionary mapping signature_key to dictionary of
      user_specified_name to Tensor representing default values.

  Returns:
    A tuple of (_AssetInfo, Graph) containing the captured assets and
    exported Graph generated from tracing the saveable_view.
  """
    resource_initializers = saveable_view.get_concrete_resource_initializers()
    exported_graph = ops.Graph()
    resource_initializer_ops = []
    with exported_graph.as_default():
        object_map, tensor_map, asset_info = saveable_view.map_resources()
        signatures = _generate_signatures(signature_functions, object_map, defaults)
    if save_custom_gradients:
        _trace_gradient_functions(exported_graph, saveable_view)
    with exported_graph.as_default():
        for resource_initializer_function in resource_initializers:
            asset_dependencies = []
            for capture in resource_initializer_function.graph.external_captures:
                asset_initializer = asset_info.asset_initializers_by_resource.get(capture, None)
                if asset_initializer is not None:
                    asset_dependencies.append(asset_initializer)
            with ops.control_dependencies(asset_dependencies):
                mapped_initializer = object_map[resource_initializer_function]
                resource_initializer_ops.append(mapped_initializer())
        resource_initializer_ops.extend(asset_info.asset_initializers_by_resource.values())
        with ops.control_dependencies(resource_initializer_ops):
            init_op = control_flow_ops.no_op()
        meta_graph_def.collection_def[constants.MAIN_OP_KEY].node_list.value.append(init_op.name)
        meta_graph_def.signature_def[constants.INIT_OP_SIGNATURE_KEY].CopyFrom(signature_def_utils.op_signature_def(init_op, constants.INIT_OP_SIGNATURE_KEY))

    def call_with_mapped_captures(function, args):
        if function in object_map:
            return object_map[function](*args)
        return saved_model_exported_concrete.ExportedConcreteFunction(function, tensor_map)(*args)
    for obj in object_map.values():
        obj._maybe_initialize_trackable()
    named_saveable_objects, registered_savers = save_util_v1.frozen_saveables_and_savers(graph_view=saveable_view.augmented_graph_view, object_map=object_map, to_graph=exported_graph, call_with_mapped_captures=call_with_mapped_captures)
    saver = functional_saver.MultiDeviceSaver.from_saveables(named_saveable_objects, registered_savers, call_with_mapped_captures)
    with exported_graph.as_default():
        saver_def = saver.to_proto()
        meta_graph_def.saver_def.CopyFrom(saver_def)
    _dependency_sorted_node_ids(saveable_view)
    graph_def, _ = exported_graph._as_graph_def(add_shapes=True, use_pybind11_proto=False)
    graph_def.library.registered_gradients.extend(saveable_view.gradient_defs)
    _verify_ops(graph_def, namespace_whitelist)
    meta_graph_def.graph_def.CopyFrom(graph_def)
    meta_graph_def.meta_info_def.tags.append(tag_constants.SERVING)
    meta_graph_def.meta_info_def.tensorflow_version = versions.__version__
    meta_graph_def.meta_info_def.tensorflow_git_version = versions.__git_version__
    meta_graph_def.meta_info_def.stripped_default_attrs = True
    meta_graph_def.meta_info_def.stripped_op_list.MergeFrom(meta_graph.stripped_op_list_for_graph(meta_graph_def.graph_def))
    meta_graph_def.asset_file_def.extend(asset_info.asset_defs)
    for signature_key, signature in signatures.items():
        meta_graph_def.signature_def[signature_key].CopyFrom(signature)
    meta_graph.strip_graph_default_valued_attrs(meta_graph_def)
    if sys.byteorder == 'big':
        utils_impl.swap_function_tensor_content(meta_graph_def, 'big', 'little')
    return (asset_info, exported_graph)