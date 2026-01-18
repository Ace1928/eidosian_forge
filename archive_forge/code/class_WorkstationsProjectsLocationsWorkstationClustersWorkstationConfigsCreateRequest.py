from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsCreateRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsCrea
  teRequest object.

  Fields:
    parent: Required. Parent resource name.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
    workstationConfig: A WorkstationConfig resource to be passed as the
      request body.
    workstationConfigId: Required. ID to use for the workstation
      configuration.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    workstationConfig = _messages.MessageField('WorkstationConfig', 3)
    workstationConfigId = _messages.StringField(4)