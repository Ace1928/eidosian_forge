from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1ProjectRepoId(_messages.Message):
    """Selects a repo using a Google Cloud Platform project ID (e.g., winged-
  cargo-31) and a repo name within that project.

  Fields:
    projectId: The ID of the project.
    repoName: The name of the repo. Leave empty for the default repo.
  """
    projectId = _messages.StringField(1)
    repoName = _messages.StringField(2)