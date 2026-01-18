from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectDataSourceDetails(_messages.Message):
    """The results of an inspect DataSource job.

  Fields:
    requestedOptions: The configuration used for this job.
    result: A summary of the outcome of this inspection job.
  """
    requestedOptions = _messages.MessageField('GooglePrivacyDlpV2RequestedOptions', 1)
    result = _messages.MessageField('GooglePrivacyDlpV2Result', 2)