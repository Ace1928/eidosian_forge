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
def _add_attributes_to_object_graph(trackable_objects, object_graph_proto, node_ids, object_names, object_map, call_with_mapped_captures, saveables_cache):
    """Create saveables/savers and corresponding protos in the object graph."""
    for checkpoint_id, (trackable, unused_object_proto) in enumerate(zip(trackable_objects, object_graph_proto.nodes)):
        assert node_ids[trackable] == checkpoint_id
    checkpoint_factory_map, unmapped_registered_savers = get_checkpoint_factories_and_keys(object_names, object_map)
    registered_savers = _add_attributes_to_object_graph_for_registered_savers(unmapped_registered_savers, object_graph_proto, node_ids, object_map)
    named_saveable_objects, feed_additions = generate_saveable_objects(checkpoint_factory_map, object_graph_proto, node_ids, object_map, call_with_mapped_captures, saveables_cache)
    return (named_saveable_objects, feed_additions, registered_savers)