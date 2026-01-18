from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadProperties(_messages.Message):
    """Properties of an underlying compute resource represented by the
  Workload.

  Fields:
    gcpProject: Output only. The service project identifier that the
      underlying cloud resource resides in. Empty for non cloud resources.
    location: Output only. The location that the underlying compute resource
      resides in (e.g us-west1).
    zone: Output only. The location that the underlying compute resource
      resides in if it is zonal (e.g us-west1-a).
  """
    gcpProject = _messages.StringField(1)
    location = _messages.StringField(2)
    zone = _messages.StringField(3)