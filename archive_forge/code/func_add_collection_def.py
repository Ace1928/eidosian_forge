import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def add_collection_def(meta_graph_def, key, graph=None, export_scope=None, exclude_nodes=None, override_contents=None):
    """Adds a collection to MetaGraphDef protocol buffer.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    key: One of the GraphKeys or user-defined string.
    graph: The `Graph` from which to get collections.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from the
      collection, or None.
    override_contents: An iterable of values to place in the collection,
      ignoring the current values (if set).
  """
    if graph and (not isinstance(graph, ops.Graph)):
        raise TypeError(f'graph must be of type Graph. Received type: {type(graph)}.')
    if not isinstance(key, str) and (not isinstance(key, bytes)):
        logging.warning('Only collections with string type keys will be serialized. This key has %s', type(key))
        return
    graph = graph or ops.get_default_graph()
    if override_contents:
        collection_list = override_contents
    else:
        collection_list = graph.get_collection(key)
    collection_list = [x for x in collection_list if _should_include_node(x, export_scope, exclude_nodes)]
    if not collection_list:
        return
    try:
        col_def = meta_graph_def.collection_def[key]
        to_proto = ops.get_to_proto_function(key)
        proto_type = ops.get_collection_proto_type(key)
        if to_proto:
            kind = 'bytes_list'
            for x in collection_list:
                proto = to_proto(x, export_scope=export_scope)
                if proto:
                    assert isinstance(proto, proto_type)
                    getattr(col_def, kind).value.append(proto.SerializeToString())
        else:
            kind = _get_kind_name(collection_list[0])
            if kind == 'node_list':
                for x in collection_list:
                    if not export_scope or x.name.startswith(export_scope):
                        getattr(col_def, kind).value.append(ops.strip_name_scope(x.name, export_scope))
            elif kind == 'bytes_list':
                getattr(col_def, kind).value.extend([compat.as_bytes(x) for x in collection_list])
            else:
                getattr(col_def, kind).value.extend([x for x in collection_list])
    except Exception as e:
        logging.warning("Issue encountered when serializing %s.\nType is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.\n%s", key, str(e))
        if key in meta_graph_def.collection_def:
            del meta_graph_def.collection_def[key]
        return