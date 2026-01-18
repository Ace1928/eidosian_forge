from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _gather_saveables(self, object_graph_tensor=None):
    serialized_tensors, feed_additions, registered_savers, graph_proto = self._gather_serialized_tensors(object_graph_tensor)
    saveables_dict = self._saveables_cache
    if saveables_dict is None:
        object_graph_tensor = serialized_tensors.pop(None)[base.OBJECT_GRAPH_PROTO_KEY]
        saveables_dict = saveable_object_util.serialized_tensors_to_saveable_cache(serialized_tensors)
    named_saveable_objects = []
    for saveable_by_name in saveables_dict.values():
        for saveables in saveable_by_name.values():
            named_saveable_objects.extend(saveables)
    named_saveable_objects.append(base.NoRestoreSaveable(tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
    return (named_saveable_objects, graph_proto, feed_additions, registered_savers)