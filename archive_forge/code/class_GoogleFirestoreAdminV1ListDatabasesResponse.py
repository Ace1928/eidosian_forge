from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ListDatabasesResponse(_messages.Message):
    """The list of databases for a project.

  Fields:
    databases: The databases in the project.
    unreachable: In the event that data about individual databases cannot be
      listed they will be recorded here. An example entry might be:
      projects/some_project/locations/some_location This can happen if the
      Cloud Region that the Database resides in is currently unavailable. In
      this case we can't fetch all the details about the database. You may be
      able to get a more detailed error message (or possibly fetch the
      resource) by sending a 'Get' request for the resource or a 'List'
      request for the specific location.
  """
    databases = _messages.MessageField('GoogleFirestoreAdminV1Database', 1, repeated=True)
    unreachable = _messages.StringField(2, repeated=True)