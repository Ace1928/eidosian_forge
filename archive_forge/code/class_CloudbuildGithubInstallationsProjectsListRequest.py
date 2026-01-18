from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildGithubInstallationsProjectsListRequest(_messages.Message):
    """A CloudbuildGithubInstallationsProjectsListRequest object.

  Fields:
    installationId: Installation ID
  """
    installationId = _messages.IntegerField(1, required=True)