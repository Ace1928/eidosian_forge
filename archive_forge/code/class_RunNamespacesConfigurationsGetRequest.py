from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesConfigurationsGetRequest(_messages.Message):
    """A RunNamespacesConfigurationsGetRequest object.

  Fields:
    name: The name of the configuration to retrieve. For Cloud Run, replace
      {namespace_id} with the project ID or number.
  """
    name = _messages.StringField(1, required=True)