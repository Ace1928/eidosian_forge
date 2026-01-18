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
class CapturableResource(base.Trackable, metaclass=_ResourceMetaclass):
    """Holds a Tensor which a tf.function can capture.

  `CapturableResource`s are discovered by traversing the graph of object
  attributes, e.g. during `tf.saved_model.save`. They are excluded from the
  scope-based tracking of `TrackableResource`; generally things that require
  initialization should inherit from `TrackableResource` instead of
  `CapturableResource` directly.
  """

    def __init__(self, device=''):
        """Initialize the `CapturableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
    """
        self._resource_handle_value = None
        self._resource_device = device
        self._self_destruction_context = context.eager_mode if context.executing_eagerly() else ops.get_default_graph().as_default

    @classmethod
    def _resource_type(cls):
        return cls.__name__

    @property
    def _destruction_context(self):
        return getattr(self, '_self_destruction_context', contextlib.suppress)

    @_destruction_context.setter
    def _destruction_context(self, destruction_context):
        self._self_destruction_context = destruction_context

    def _create_resource(self):
        """A function that creates a resource handle."""
        raise NotImplementedError('TrackableResource._create_resource not implemented.')

    @property
    def _resource_handle(self):
        return self._resource_handle_value

    @_resource_handle.setter
    def _resource_handle(self, value):
        if isinstance(value, (tensor.Tensor, ops.EagerTensor)):
            value._parent_trackable = weakref.ref(self)
        self._resource_handle_value = value

    def _initialize(self):
        """A function that initializes the resource. Optional."""
        pass

    def _destroy_resource(self):
        """A function that destroys the resource. Optional."""
        pass

    @property
    def resource_handle(self):
        """Returns the resource handle associated with this Resource."""
        if self._resource_handle is None:
            with ops.device(self._resource_device):
                self._resource_handle = self._create_resource()
        return self._resource_handle

    def _export_to_saved_model_graph(self, object_map, tensor_map, **unused_kwargs):
        """For implementing `Trackable`."""
        new_obj = copy.copy(self)
        with ops.device(self._resource_device):
            new_resource = new_obj._create_resource()
        new_obj._resource_handle = new_resource
        object_map[self] = new_obj
        tensor_map[self.resource_handle] = new_resource
        return [self.resource_handle]

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        children = super()._trackable_children(save_type, **kwargs)
        if save_type == 'savedmodel':

            @def_function.function(input_signature=[], autograph=False)
            def _creator():
                resource = self._create_resource()
                return resource

            @def_function.function(input_signature=[], autograph=False)
            def _initializer():
                self._initialize()
                return 1

            @def_function.function(input_signature=[], autograph=False)
            def _destroyer():
                self._destroy_resource()
                return 1
            children.update({'_create_resource': _creator, '_initialize': _initializer, '_destroy_resource': _destroyer})
        return children

    def __del__(self):
        try:
            with self._destruction_context():
                self._destroy_resource()
        except Exception:
            pass