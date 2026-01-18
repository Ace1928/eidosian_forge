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
def _get_tensors_from_trackable(trackable_data: _TrackableData, call_with_mapped_captures: Union[Callable[..., Any], None], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Dict[str, Any]:
    """Gets tensors to serialize from a Trackable."""
    trackable = trackable_data.object_to_save
    save_fn = trackable._serialize_to_tensors
    if call_with_mapped_captures and isinstance(save_fn, core.ConcreteFunction):
        ret_tensor_dict = call_with_mapped_captures(save_fn, [])
    else:
        ret_tensor_dict = save_fn()
    tensor_dict = {}
    for tensor_name, maybe_tensor in ret_tensor_dict.items():
        local_name = trackable_utils.escape_local_name(tensor_name)
        checkpoint_key = trackable_utils.checkpoint_key(trackable_data.object_name, local_name)
        tensor_dict[checkpoint_key] = maybe_tensor
        if isinstance(maybe_tensor, saveable_object_lib.SaveSpec):
            maybe_tensor.name = checkpoint_key
            maybe_tensor.slice_spec = ''
        if object_graph_proto is not None:
            object_graph_proto.nodes[trackable_data.node_id].attributes.add(name=local_name, checkpoint_key=checkpoint_key, full_name=util.get_full_name(trackable))
    return tensor_dict