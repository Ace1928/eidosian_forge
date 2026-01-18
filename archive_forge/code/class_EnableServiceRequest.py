from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableServiceRequest(_messages.Message):
    """Request message for EnableService method.

  Fields:
    consumerId: The identity of consumer resource which service enablement
      will be applied to.  The Google Service Management implementation
      accepts the following forms: "project:<project_id>",
      "project_number:<project_number>".  Note: this is made compatible with
      google.api.servicecontrol.v1.Operation.consumer_id.
  """
    consumerId = _messages.StringField(1)