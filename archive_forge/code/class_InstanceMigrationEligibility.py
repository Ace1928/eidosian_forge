from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceMigrationEligibility(_messages.Message):
    """InstanceMigrationEligibility represents the feasibility information of a
  migration from UmN to WbI.

  Enums:
    ErrorsValueListEntryValuesEnum:
    WarningsValueListEntryValuesEnum:

  Fields:
    errors: Output only. Certain configurations make the UmN ineligible for an
      automatic migration. A manual migration is required.
    warnings: Output only. Certain configurations will be defaulted during the
      migration.
  """

    class ErrorsValueListEntryValuesEnum(_messages.Enum):
        """ErrorsValueListEntryValuesEnum enum type.

    Values:
      ERROR_UNSPECIFIED: Default type.
      DATAPROC_HUB: The UmN uses Dataproc Hub and cannot be migrated.
    """
        ERROR_UNSPECIFIED = 0
        DATAPROC_HUB = 1

    class WarningsValueListEntryValuesEnum(_messages.Enum):
        """WarningsValueListEntryValuesEnum enum type.

    Values:
      WARNING_UNSPECIFIED: Default type.
      UNSUPPORTED_MACHINE_TYPE: The UmN uses an machine type that's
        unsupported in WbI. It will be migrated with the default machine type
        e2-standard-4. Users can change the machine type after the migration.
      UNSUPPORTED_ACCELERATOR_TYPE: The UmN uses an accelerator type that's
        unsupported in WbI. It will be migrated without an accelerator. User
        can attach an accelerator after the migration.
      UNSUPPORTED_OS: The UmN uses an operating system that's unsupported in
        WbI (e.g. Debian 10, Ubuntu). It will be replaced with Debian 11 in
        WbI.
      NO_REMOVE_DATA_DISK: This UmN is configured with no_remove_data_disk,
        which is no longer available in WbI.
      GCS_BACKUP: This UmN is configured with the Cloud Storage backup
        feature, which is no longer available in WbI.
      POST_STARTUP_SCRIPT: This UmN is configured with a post startup script.
        Please optionally provide the `post_startup_script_option` for the
        migration.
    """
        WARNING_UNSPECIFIED = 0
        UNSUPPORTED_MACHINE_TYPE = 1
        UNSUPPORTED_ACCELERATOR_TYPE = 2
        UNSUPPORTED_OS = 3
        NO_REMOVE_DATA_DISK = 4
        GCS_BACKUP = 5
        POST_STARTUP_SCRIPT = 6
    errors = _messages.EnumField('ErrorsValueListEntryValuesEnum', 1, repeated=True)
    warnings = _messages.EnumField('WarningsValueListEntryValuesEnum', 2, repeated=True)