from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
@override(Schedule)
def _tf_value_op(self, t: TensorType) -> TensorType:
    return tf.constant(self._v)