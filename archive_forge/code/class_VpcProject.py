from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcProject(_messages.Message):
    """Project detail of the VPC network.

  Fields:
    projectId: The project of the VPC to connect to. If not specified, it is
      the same as the cluster project.
  """
    projectId = _messages.StringField(1)