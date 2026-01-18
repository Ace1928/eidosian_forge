from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1ApprovalConfig(_messages.Message):
    """ApprovalConfig describes configuration for manual approval of a build.

  Fields:
    approvalRequired: Whether or not approval is needed. If this is set on a
      build, it will become pending when created, and will need to be
      explicitly approved to start.
  """
    approvalRequired = _messages.BooleanField(1)