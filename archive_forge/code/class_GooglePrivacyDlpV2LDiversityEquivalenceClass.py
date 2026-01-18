from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LDiversityEquivalenceClass(_messages.Message):
    """The set of columns' values that share the same ldiversity value.

  Fields:
    equivalenceClassSize: Size of the k-anonymity equivalence class.
    numDistinctSensitiveValues: Number of distinct sensitive values in this
      equivalence class.
    quasiIdsValues: Quasi-identifier values defining the k-anonymity
      equivalence class. The order is always the same as the original request.
    topSensitiveValues: Estimated frequencies of top sensitive values.
  """
    equivalenceClassSize = _messages.IntegerField(1)
    numDistinctSensitiveValues = _messages.IntegerField(2)
    quasiIdsValues = _messages.MessageField('GooglePrivacyDlpV2Value', 3, repeated=True)
    topSensitiveValues = _messages.MessageField('GooglePrivacyDlpV2ValueFrequency', 4, repeated=True)