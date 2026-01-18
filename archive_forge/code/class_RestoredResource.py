import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
class RestoredResource(TrackableResource):
    """Restored SavedResource."""

    def __init__(self, device=''):
        super().__init__(device=device)

    @classmethod
    def _deserialize_from_proto(cls, object_proto, dependencies, **unused_kwargs):
        obj = cls(device=object_proto.resource.device)
        resource_creator = dependencies.get('_create_resource')
        if resource_creator is not None:
            obj._create_resource = resource_creator
        return obj

    def _add_trackable_child(self, name, value):
        setattr(self, name, value)
        if isinstance(value, base.Trackable) and (not isinstance(value, def_function.Function)):
            self._track_trackable(value, name)