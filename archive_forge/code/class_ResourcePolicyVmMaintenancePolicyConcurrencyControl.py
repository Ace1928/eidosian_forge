from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyVmMaintenancePolicyConcurrencyControl(_messages.Message):
    """A concurrency control configuration. Defines a group config that, when
  attached to an instance, recognizes that instance as part of a group of
  instances where only up the concurrency_limit of instances in that group can
  undergo simultaneous maintenance. For more information: go/concurrency-
  control-design-doc

  Fields:
    concurrencyLimit: A integer attribute.
  """
    concurrencyLimit = _messages.IntegerField(1, variant=_messages.Variant.INT32)