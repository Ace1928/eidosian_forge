from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BoostConfig(_messages.Message):
    """A configuration that workstations can boost to.

  Fields:
    accelerators: Optional. A list of the type and count of accelerator cards
      attached to the boost instance. Defaults to `none`.
    id: Optional. Required. The id to be used for the boost config.
    machineType: Optional. The type of machine that boosted VM instances will
      use-for example, `e2-standard-4`. For more information about machine
      types that Cloud Workstations supports, see the list of [available
      machine types](https://cloud.google.com/workstations/docs/available-
      machine-types). Defaults to `e2-standard-4`.
    poolSize: Optional. The number of boost VMs that the system should keep
      idle so that workstations can be boosted quickly. Defaults to `0`.
  """
    accelerators = _messages.MessageField('Accelerator', 1, repeated=True)
    id = _messages.StringField(2)
    machineType = _messages.StringField(3)
    poolSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)