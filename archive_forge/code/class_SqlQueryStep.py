from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlQueryStep(_messages.Message):
    """A query step defined in raw SQL.

  Fields:
    parameters: Optional. Parameters to be injected into the query at
      execution time.
    queryRestriction: Optional. Restrictions being requested, e.g. timerange
      restrictions.
    sqlQuery: Required. A query string, following the BigQuery SQL query
      syntax. The FROM clause should specify a fully qualified log view
      corresponding to the log view in the resource_names in dot separated
      format like PROJECT_ID.LOCATION_ID.BUCKET_ID.VIEW_ID.For example: SELECT
      count(*) FROM my_project.us.my_bucket._AllLogs;If any of the dot
      separated components have special characters, that component needs to be
      escaped separately like the following example:SELECT count(*) FROM
      company.com:abc.us.my-bucket._AllLogs;
  """
    parameters = _messages.MessageField('QueryParameter', 1, repeated=True)
    queryRestriction = _messages.MessageField('QueryRestriction', 2)
    sqlQuery = _messages.StringField(3)