from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1GitSourceContext(_messages.Message):
    """A GitSourceContext denotes a particular revision in a third party Git
  repository (e.g., GitHub).

  Fields:
    revisionId: Required. Git commit hash.
    url: Git repository URL.
  """
    revisionId = _messages.StringField(1)
    url = _messages.StringField(2)