from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1ViolationRemediationInstructions(_messages.Message):
    """Instructions to remediate violation

  Fields:
    consoleInstructions: Remediation instructions to resolve violation via
      cloud console
    gcloudInstructions: Remediation instructions to resolve violation via
      gcloud cli
  """
    consoleInstructions = _messages.MessageField('GoogleCloudAssuredworkloadsV1ViolationRemediationInstructionsConsole', 1)
    gcloudInstructions = _messages.MessageField('GoogleCloudAssuredworkloadsV1ViolationRemediationInstructionsGcloud', 2)