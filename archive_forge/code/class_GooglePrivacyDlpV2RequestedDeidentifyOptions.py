from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RequestedDeidentifyOptions(_messages.Message):
    """De-identification options.

  Fields:
    snapshotDeidentifyTemplate: Snapshot of the state of the
      `DeidentifyTemplate` from the Deidentify action at the time this job was
      run.
    snapshotImageRedactTemplate: Snapshot of the state of the image
      transformation `DeidentifyTemplate` from the `Deidentify` action at the
      time this job was run.
    snapshotStructuredDeidentifyTemplate: Snapshot of the state of the
      structured `DeidentifyTemplate` from the `Deidentify` action at the time
      this job was run.
  """
    snapshotDeidentifyTemplate = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 1)
    snapshotImageRedactTemplate = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 2)
    snapshotStructuredDeidentifyTemplate = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 3)