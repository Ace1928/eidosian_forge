from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KAnonymityEquivalenceClass(_messages.Message):
    """The set of columns' values that share the same ldiversity value

  Fields:
    equivalenceClassSize: Size of the equivalence class, for example number of
      rows with the above set of values.
    quasiIdsValues: Set of values defining the equivalence class. One value
      per quasi-identifier column in the original KAnonymity metric message.
      The order is always the same as the original request.
  """
    equivalenceClassSize = _messages.IntegerField(1)
    quasiIdsValues = _messages.MessageField('GooglePrivacyDlpV2Value', 2, repeated=True)