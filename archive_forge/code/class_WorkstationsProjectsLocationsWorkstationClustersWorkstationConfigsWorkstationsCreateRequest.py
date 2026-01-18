from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsCreateRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsCreateRequest object.

  Fields:
    parent: Required. Parent resource name.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
    workstation: A Workstation resource to be passed as the request body.
    workstationId: Required. ID to use for the workstation.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    workstation = _messages.MessageField('Workstation', 3)
    workstationId = _messages.StringField(4)