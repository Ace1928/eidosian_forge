from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNfsSharesPatchRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNfsSharesPatchRequest object.

  Fields:
    name: Immutable. The name of the NFS share.
    nfsShare: A NfsShare resource to be passed as the request body.
    updateMask: The list of fields to update. The only currently supported
      fields are: `labels` `allowed_clients`
  """
    name = _messages.StringField(1, required=True)
    nfsShare = _messages.MessageField('NfsShare', 2)
    updateMask = _messages.StringField(3)