from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunApiV1NamespacesGetRequest(_messages.Message):
    """A RunApiV1NamespacesGetRequest object.

  Fields:
    name: Required. The name of the namespace being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    name = _messages.StringField(1, required=True)