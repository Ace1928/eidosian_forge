from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsCommitFunctionUpgradeRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsCommitFunctionUpgradeRequest
  object.

  Fields:
    commitFunctionUpgradeRequest: A CommitFunctionUpgradeRequest resource to
      be passed as the request body.
    name: Required. The name of the function for which upgrade should be
      finalized.
  """
    commitFunctionUpgradeRequest = _messages.MessageField('CommitFunctionUpgradeRequest', 1)
    name = _messages.StringField(2, required=True)