from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsProjectsLocationsNamespacesCreateRequest(_messages.Message):
    """A AnthoseventsProjectsLocationsNamespacesCreateRequest object.

  Fields:
    namespace: A Namespace resource to be passed as the request body.
    parent: Required. The project ID or project number in which this namespace
      should be created.
  """
    namespace = _messages.MessageField('Namespace', 1)
    parent = _messages.StringField(2, required=True)