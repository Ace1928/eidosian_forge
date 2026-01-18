from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInterconnectsGetMacsecConfigRequest(_messages.Message):
    """A ComputeInterconnectsGetMacsecConfigRequest object.

  Fields:
    interconnect: Name of the interconnect resource to query.
    project: Project ID for this request.
  """
    interconnect = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)