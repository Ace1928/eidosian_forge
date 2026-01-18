from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsPatchRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsPatchRequest object.

  Fields:
    allowMissing: Optional. If set and the workstation configuration is not
      found, a new workstation configuration is created. In this situation,
      update_mask is ignored.
    name: Identifier. Full name of this workstation.
    updateMask: Required. Mask specifying which fields in the workstation
      configuration should be updated.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
    workstation: A Workstation resource to be passed as the request body.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)
    workstation = _messages.MessageField('Workstation', 5)