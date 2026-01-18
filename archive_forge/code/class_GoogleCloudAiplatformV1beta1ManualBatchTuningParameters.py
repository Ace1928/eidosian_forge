from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ManualBatchTuningParameters(_messages.Message):
    """Manual batch tuning parameters.

  Fields:
    batchSize: Immutable. The number of the records (e.g. instances) of the
      operation given in each batch to a machine replica. Machine type, and
      size of a single record should be considered when setting this
      parameter, higher value speeds up the batch operation's execution, but
      too high value will result in a whole batch not fitting in a machine's
      memory, and the whole operation will fail. The default value is 64.
  """
    batchSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)