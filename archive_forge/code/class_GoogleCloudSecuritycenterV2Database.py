from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Database(_messages.Message):
    """Represents database access information, such as queries. A database may
  be a sub-resource of an instance (as in the case of Cloud SQL instances or
  Cloud Spanner instances), or the database instance itself. Some database
  resources might not have the [full resource
  name](https://google.aip.dev/122#full-resource-names) populated because
  these resource types, such as Cloud SQL databases, are not yet supported by
  Cloud Asset Inventory. In these cases only the display name is provided.

  Fields:
    displayName: The human-readable name of the database that the user
      connected to.
    grantees: The target usernames, roles, or groups of an SQL privilege
      grant, which is not an IAM policy change.
    name: Some database resources may not have the [full resource
      name](https://google.aip.dev/122#full-resource-names) populated because
      these resource types are not yet supported by Cloud Asset Inventory
      (e.g. Cloud SQL databases). In these cases only the display name will be
      provided. The [full resource name](https://google.aip.dev/122#full-
      resource-names) of the database that the user connected to, if it is
      supported by Cloud Asset Inventory.
    query: The SQL statement that is associated with the database access.
    userName: The username used to connect to the database. The username might
      not be an IAM principal and does not have a set format.
    version: The version of the database, for example, POSTGRES_14. See [the
      complete list](https://cloud.google.com/sql/docs/mysql/admin-
      api/rest/v1/SqlDatabaseVersion).
  """
    displayName = _messages.StringField(1)
    grantees = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    query = _messages.StringField(4)
    userName = _messages.StringField(5)
    version = _messages.StringField(6)