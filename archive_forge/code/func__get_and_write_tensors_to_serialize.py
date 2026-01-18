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
def _get_and_write_tensors_to_serialize(tensor_trackables: List[_TrackableData], node_ids: Dict[base.Trackable, int], call_with_mapped_captures: Union[Callable[..., Any], None], cache: Union[Dict[base.Trackable, any], None], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Dict[base.Trackable, Any]:
    """Creates dictionary of tensors to checkpoint, and updates the proto."""
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    for td in tensor_trackables:
        if cache is not None and td.object_to_save in cache:
            trackable, tensor_dict, object_proto = cache[td.object_to_save]
            serialized_tensors[trackable] = tensor_dict
            object_graph_proto.nodes[td.node_id].attributes.MergeFrom(object_proto)
            continue
        legacy_name = saveable_compat.get_saveable_name(td.object_to_save) or ''
        if not saveable_object_util.trackable_has_serialize_to_tensor(td.object_to_save) or legacy_name:
            trackable, tensor_dict = _get_tensors_from_legacy_saveable(td, node_ids, call_with_mapped_captures, object_graph_proto)
        else:
            tensor_dict = _get_tensors_from_trackable(td, call_with_mapped_captures, object_graph_proto)
            trackable = td.object_to_save
        serialized_tensors[trackable] = tensor_dict
        if cache is not None and td.object_to_save not in cache:
            cache[td.object_to_save] = (trackable, tensor_dict, object_graph_proto.nodes[td.node_id].attributes)
    return serialized_tensors