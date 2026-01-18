from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomTargetSkaffoldActions(_messages.Message):
    """CustomTargetSkaffoldActions represents the `CustomTargetType`
  configuration using Skaffold custom actions.

  Fields:
    deployAction: Required. The Skaffold custom action responsible for deploy
      operations.
    includeSkaffoldModules: Optional. List of Skaffold modules Cloud Deploy
      will include in the Skaffold Config as required before performing
      diagnose.
    renderAction: Optional. The Skaffold custom action responsible for render
      operations. If not provided then Cloud Deploy will perform the render
      operations via `skaffold render`.
  """
    deployAction = _messages.StringField(1)
    includeSkaffoldModules = _messages.MessageField('SkaffoldModules', 2, repeated=True)
    renderAction = _messages.StringField(3)