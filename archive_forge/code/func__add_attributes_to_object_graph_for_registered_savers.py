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
def _add_attributes_to_object_graph_for_registered_savers(unmapped_registered_savers, object_graph_proto, node_ids, object_map):
    """Fills the object graph proto with data about the registered savers."""
    registered_savers = collections.defaultdict(dict)
    for saver_name, trackables in unmapped_registered_savers.items():
        for object_name, trackable in trackables.items():
            object_proto = object_graph_proto.nodes[node_ids[trackable]]
            object_proto.registered_saver.name = saver_name
            object_proto.registered_saver.object_name = object_name
            object_to_save = util.get_mapped_trackable(trackable, object_map)
            registered_savers[saver_name][object_name] = object_to_save
    return registered_savers