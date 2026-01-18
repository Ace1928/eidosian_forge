from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsDomainMappingsGetRequest(_messages.Message):
    """A AppengineAppsDomainMappingsGetRequest object.

  Fields:
    name: Name of the resource requested. Example:
      apps/myapp/domainMappings/example.com.
  """
    name = _messages.StringField(1, required=True)