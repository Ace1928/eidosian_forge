import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def _fill_object_graph_proto(graph_view, trackable_objects, node_ids, slot_variables):
    """Name non-slot `Trackable`s and add them to `object_graph_proto`."""
    object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
    for checkpoint_id, trackable in enumerate(trackable_objects):
        assert node_ids[trackable] == checkpoint_id
        object_proto = object_graph_proto.nodes.add(slot_variables=slot_variables.get(trackable, ()))
        for child in graph_view.list_children(trackable):
            object_proto.children.add(node_id=node_ids[child.ref], local_name=child.name)
    return object_graph_proto