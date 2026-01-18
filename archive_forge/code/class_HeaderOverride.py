from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeaderOverride(_messages.Message):
    """Wraps the Header object.

  Fields:
    header: Header embodying a key and a value. Do not put business sensitive
      or personally identifying data in the HTTP Header Override Configuration
      or other similar fields in accordance with Section 12 (Resource Fields)
      of the [Service Specific Terms](https://cloud.google.com/terms/service-
      terms).
  """
    header = _messages.MessageField('Header', 1)