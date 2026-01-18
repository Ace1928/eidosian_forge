from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionProperty(_messages.Message):
    """A connection-level property to customize query behavior. Under JDBC,
  these correspond directly to connection properties passed to the
  DriverManager. Under ODBC, these correspond to properties in the connection
  string. Currently supported connection properties: * **dataset_project_id**:
  represents the default project for datasets that are used in the query.
  Setting the system variable `@@dataset_project_id` achieves the same
  behavior. For more information about system variables, see:
  https://cloud.google.com/bigquery/docs/reference/system-variables *
  **time_zone**: represents the default timezone used to run the query. *
  **session_id**: associates the query with a given session. *
  **query_label**: associates the query with a given job label. If set, all
  subsequent queries in a script or session will have this label. For the
  format in which a you can specify a query label, see labels in the
  JobConfiguration resource type: https://cloud.google.com/bigquery/docs/refer
  ence/rest/v2/Job#jobconfiguration Additional properties are allowed, but
  ignored. Specifying multiple connection properties with the same key returns
  an error.

  Fields:
    key: The key of the property to set.
    value: The value of the property to set.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)