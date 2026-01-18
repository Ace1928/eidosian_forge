from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackApiDeploymentRequest(_messages.Message):
    """Request message for RollbackApiDeployment.

  Fields:
    revisionId: Required. The revision ID to roll back to. It must be a
      revision of the same deployment. Example: `c7cfa2a8`
  """
    revisionId = _messages.StringField(1)