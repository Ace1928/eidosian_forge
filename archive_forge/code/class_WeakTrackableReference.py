import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
class WeakTrackableReference(TrackableReference):
    """TrackableReference that stores weak references."""
    __slots__ = ()

    def __init__(self, name, ref):
        if not isinstance(self, weakref.ref):
            ref = weakref.ref(ref)
        super(WeakTrackableReference, self).__init__(name=name, ref=ref)

    @property
    def ref(self):
        return self._ref()