from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersPatchRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersPatchRequest object.

  Fields:
    allowMissing: Optional. If set, and the workstation cluster is not found,
      a new workstation cluster will be created. In this situation,
      update_mask is ignored.
    name: Identifier. Full name of this workstation cluster.
    updateMask: Required. Mask that specifies which fields in the workstation
      cluster should be updated.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
    workstationCluster: A WorkstationCluster resource to be passed as the
      request body.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)
    workstationCluster = _messages.MessageField('WorkstationCluster', 5)