from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1EnvironmentVersionConfig(_messages.Message):
    """Configuration for the version.

  Fields:
    version: Required. Format: projects//locations//agents//flows//versions/.
  """
    version = _messages.StringField(1)