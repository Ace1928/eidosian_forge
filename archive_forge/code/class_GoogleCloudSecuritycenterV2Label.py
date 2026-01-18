from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Label(_messages.Message):
    """Represents a generic name-value label. A label has separate name and
  value fields to support filtering with the `contains()` function. For more
  information, see [Filtering on array-type
  fields](https://cloud.google.com/security-command-center/docs/how-to-api-
  list-findings#array-contains-filtering).

  Fields:
    name: Name of the label.
    value: Value that corresponds to the label's name.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)