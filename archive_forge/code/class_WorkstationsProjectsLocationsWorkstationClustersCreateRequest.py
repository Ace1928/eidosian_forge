from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersCreateRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersCreateRequest object.

  Fields:
    parent: Required. Parent resource name.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
    workstationCluster: A WorkstationCluster resource to be passed as the
      request body.
    workstationClusterId: Required. ID to use for the workstation cluster.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    workstationCluster = _messages.MessageField('WorkstationCluster', 3)
    workstationClusterId = _messages.StringField(4)