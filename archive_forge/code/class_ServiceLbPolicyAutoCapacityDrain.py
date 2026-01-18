from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLbPolicyAutoCapacityDrain(_messages.Message):
    """Option to specify if an unhealthy IG/NEG should be considered for global
  load balancing and traffic routing.

  Fields:
    enable: Optional. If set to 'True', an unhealthy IG/NEG will be set as
      drained. - An IG/NEG is considered unhealthy if less than 25% of the
      instances/endpoints in the IG/NEG are healthy. - This option will never
      result in draining more than 50% of the configured IGs/NEGs for the
      Backend Service.
  """
    enable = _messages.BooleanField(1)