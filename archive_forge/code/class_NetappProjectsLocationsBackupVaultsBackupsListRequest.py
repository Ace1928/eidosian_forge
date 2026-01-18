from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsBackupsListRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsBackupsListRequest object.

  Fields:
    filter: The standard list filter. If specified, backups will be returned
      based on the attribute name that matches the filter expression. If
      empty, then no backups are filtered out. See https://google.aip.dev/160
    orderBy: Sort results. Supported values are "name", "name desc" or ""
      (unsorted).
    pageSize: The maximum number of items to return. The service may return
      fewer than this value. The maximum value is 1000; values above 1000 will
      be coerced to 1000.
    pageToken: The next_page_token value to use if there are additional
      results to retrieve for this list request.
    parent: Required. The backupVault for which to retrieve backup
      information, in the format `projects/{project_id}/locations/{location}/b
      ackupVaults/{backup_vault_id}`. To retrieve backup information for all
      locations, use "-" for the `{location}` value. To retrieve backup
      information for all backupVaults, use "-" for the `{backup_vault_id}`
      value. To retrieve backup information for a volume, use "-" for the
      `{backup_vault_id}` value and specify volume full name with the filter.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)