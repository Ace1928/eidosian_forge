from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RequestedOptions(_messages.Message):
    """Snapshot of the inspection configuration.

  Fields:
    jobConfig: Inspect config.
    snapshotInspectTemplate: If run with an InspectTemplate, a snapshot of its
      state at the time of this run.
  """
    jobConfig = _messages.MessageField('GooglePrivacyDlpV2InspectJobConfig', 1)
    snapshotInspectTemplate = _messages.MessageField('GooglePrivacyDlpV2InspectTemplate', 2)