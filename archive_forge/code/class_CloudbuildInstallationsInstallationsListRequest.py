from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildInstallationsInstallationsListRequest(_messages.Message):
    """A CloudbuildInstallationsInstallationsListRequest object.

  Fields:
    installationId: Installation ID
  """
    installationId = _messages.IntegerField(1, required=True)