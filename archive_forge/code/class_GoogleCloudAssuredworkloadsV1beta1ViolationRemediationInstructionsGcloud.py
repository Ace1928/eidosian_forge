from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1ViolationRemediationInstructionsGcloud(_messages.Message):
    """Remediation instructions to resolve violation via gcloud cli

  Fields:
    additionalLinks: Additional urls for more information about steps
    gcloudCommands: Gcloud command to resolve violation
    steps: Steps to resolve violation via gcloud cli
  """
    additionalLinks = _messages.StringField(1, repeated=True)
    gcloudCommands = _messages.StringField(2, repeated=True)
    steps = _messages.StringField(3, repeated=True)