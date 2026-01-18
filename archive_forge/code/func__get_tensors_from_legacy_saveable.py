import collections
from typing import Any, Callable, List, Optional, Tuple, Mapping, Union, Dict
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import save_util_v1
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
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def _get_tensors_from_legacy_saveable(trackable_data: _TrackableData, node_ids: Dict[base.Trackable, int], call_with_mapped_captures: Callable[..., Any], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Tuple[base.Trackable, Dict[str, Any]]:
    """Gets tensors to serialize from a Trackable with legacy SaveableObjects."""
    object_names = object_identity.ObjectIdentityDictionary()
    object_names[trackable_data.trackable] = trackable_data.object_name
    object_map = object_identity.ObjectIdentityDictionary()
    object_map[trackable_data.trackable] = trackable_data.object_to_save
    checkpoint_factory_map, _ = save_util_v1.get_checkpoint_factories_and_keys(object_names, object_map)
    named_saveable_objects, _ = save_util_v1.generate_saveable_objects(checkpoint_factory_map, object_graph_proto, node_ids, object_map, call_with_mapped_captures, saveables_cache=None)
    trackable = saveable_object_util.SaveableCompatibilityConverter(trackable_data.object_to_save, named_saveable_objects)
    return (trackable, trackable._serialize_to_tensors())