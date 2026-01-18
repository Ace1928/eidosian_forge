from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsCreateRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsCreateRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    connectionId: Required. Identifier to assign to the Connection. Must be
      unique within scope of the parent resource.
    parent: Required. Parent resource of the Connection, of the form:
      `projects/*/locations/*`
  """
    connection = _messages.MessageField('Connection', 1)
    connectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)