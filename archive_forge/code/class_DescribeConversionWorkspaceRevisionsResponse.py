from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DescribeConversionWorkspaceRevisionsResponse(_messages.Message):
    """Response message for 'DescribeConversionWorkspaceRevisions' request.

  Fields:
    revisions: The list of conversion workspace revisions.
  """
    revisions = _messages.MessageField('ConversionWorkspace', 1, repeated=True)