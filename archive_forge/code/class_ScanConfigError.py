from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ScanConfigError(_messages.Message):
    """Defines a custom error message used by CreateScanConfig and
  UpdateScanConfig APIs when scan configuration validation fails. It is also
  reported as part of a ScanRunErrorTrace message if scan validation fails due
  to a scan configuration error.

  Enums:
    CodeValueValuesEnum: Indicates the reason code for a configuration
      failure.

  Fields:
    code: Indicates the reason code for a configuration failure.
    fieldName: Indicates the full name of the ScanConfig field that triggers
      this error, for example "scan_config.max_qps". This field is provided
      for troubleshooting purposes only and its actual value can change in the
      future.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Indicates the reason code for a configuration failure.

    Values:
      CODE_UNSPECIFIED: There is no error.
      OK: There is no error.
      INTERNAL_ERROR: Indicates an internal server error. Please DO NOT USE
        THIS ERROR CODE unless the root cause is truly unknown.
      APPENGINE_API_BACKEND_ERROR: One of the seed URLs is an App Engine URL
        but we cannot validate the scan settings due to an App Engine API
        backend error.
      APPENGINE_API_NOT_ACCESSIBLE: One of the seed URLs is an App Engine URL
        but we cannot access the App Engine API to validate scan settings.
      APPENGINE_DEFAULT_HOST_MISSING: One of the seed URLs is an App Engine
        URL but the Default Host of the App Engine is not set.
      CANNOT_USE_GOOGLE_COM_ACCOUNT: Google corporate accounts can not be used
        for scanning.
      CANNOT_USE_OWNER_ACCOUNT: The account of the scan creator can not be
        used for scanning.
      COMPUTE_API_BACKEND_ERROR: This scan targets Compute Engine, but we
        cannot validate scan settings due to a Compute Engine API backend
        error.
      COMPUTE_API_NOT_ACCESSIBLE: This scan targets Compute Engine, but we
        cannot access the Compute Engine API to validate the scan settings.
      CUSTOM_LOGIN_URL_DOES_NOT_BELONG_TO_CURRENT_PROJECT: The Custom Login
        URL does not belong to the current project.
      CUSTOM_LOGIN_URL_MALFORMED: The Custom Login URL is malformed (can not
        be parsed).
      CUSTOM_LOGIN_URL_MAPPED_TO_NON_ROUTABLE_ADDRESS: The Custom Login URL is
        mapped to a non-routable IP address in DNS.
      CUSTOM_LOGIN_URL_MAPPED_TO_UNRESERVED_ADDRESS: The Custom Login URL is
        mapped to an IP address which is not reserved for the current project.
      CUSTOM_LOGIN_URL_HAS_NON_ROUTABLE_IP_ADDRESS: The Custom Login URL has a
        non-routable IP address.
      CUSTOM_LOGIN_URL_HAS_UNRESERVED_IP_ADDRESS: The Custom Login URL has an
        IP address which is not reserved for the current project.
      DUPLICATE_SCAN_NAME: Another scan with the same name (case-sensitive)
        already exists.
      INVALID_FIELD_VALUE: A field is set to an invalid value.
      FAILED_TO_AUTHENTICATE_TO_TARGET: There was an error trying to
        authenticate to the scan target.
      FINDING_TYPE_UNSPECIFIED: Finding type value is not specified in the
        list findings request.
      FORBIDDEN_TO_SCAN_COMPUTE: Scan targets Compute Engine, yet current
        project was not whitelisted for Google Compute Engine Scanning Alpha
        access.
      FORBIDDEN_UPDATE_TO_MANAGED_SCAN: User tries to update managed scan
      MALFORMED_FILTER: The supplied filter is malformed. For example, it can
        not be parsed, does not have a filter type in expression, or the same
        filter type appears more than once.
      MALFORMED_RESOURCE_NAME: The supplied resource name is malformed (can
        not be parsed).
      PROJECT_INACTIVE: The current project is not in an active state.
      REQUIRED_FIELD: A required field is not set.
      RESOURCE_NAME_INCONSISTENT: Project id, scanconfig id, scanrun id, or
        finding id are not consistent with each other in resource name.
      SCAN_ALREADY_RUNNING: The scan being requested to start is already
        running.
      SCAN_NOT_RUNNING: The scan that was requested to be stopped is not
        running.
      SEED_URL_DOES_NOT_BELONG_TO_CURRENT_PROJECT: One of the seed URLs does
        not belong to the current project.
      SEED_URL_MALFORMED: One of the seed URLs is malformed (can not be
        parsed).
      SEED_URL_MAPPED_TO_NON_ROUTABLE_ADDRESS: One of the seed URLs is mapped
        to a non-routable IP address in DNS.
      SEED_URL_MAPPED_TO_UNRESERVED_ADDRESS: One of the seed URLs is mapped to
        an IP address which is not reserved for the current project.
      SEED_URL_HAS_NON_ROUTABLE_IP_ADDRESS: One of the seed URLs has on-
        routable IP address.
      SEED_URL_HAS_UNRESERVED_IP_ADDRESS: One of the seed URLs has an IP
        address that is not reserved for the current project.
      SERVICE_ACCOUNT_NOT_CONFIGURED: The Web Security Scanner service account
        is not configured under the project.
      TOO_MANY_SCANS: A project has reached the maximum number of scans.
      UNABLE_TO_RESOLVE_PROJECT_INFO: Resolving the details of the current
        project fails.
      UNSUPPORTED_BLACKLIST_PATTERN_FORMAT: One or more blacklist patterns
        were in the wrong format.
      UNSUPPORTED_FILTER: The supplied filter is not supported.
      UNSUPPORTED_FINDING_TYPE: The supplied finding type is not supported.
        For example, we do not provide findings of the given finding type.
      UNSUPPORTED_URL_SCHEME: The URL scheme of one or more of the supplied
        URLs is not supported.
      CLOUD_ASSET_INVENTORY_ASSET_NOT_FOUND: CAI is not able to list assets.
    """
        CODE_UNSPECIFIED = 0
        OK = 1
        INTERNAL_ERROR = 2
        APPENGINE_API_BACKEND_ERROR = 3
        APPENGINE_API_NOT_ACCESSIBLE = 4
        APPENGINE_DEFAULT_HOST_MISSING = 5
        CANNOT_USE_GOOGLE_COM_ACCOUNT = 6
        CANNOT_USE_OWNER_ACCOUNT = 7
        COMPUTE_API_BACKEND_ERROR = 8
        COMPUTE_API_NOT_ACCESSIBLE = 9
        CUSTOM_LOGIN_URL_DOES_NOT_BELONG_TO_CURRENT_PROJECT = 10
        CUSTOM_LOGIN_URL_MALFORMED = 11
        CUSTOM_LOGIN_URL_MAPPED_TO_NON_ROUTABLE_ADDRESS = 12
        CUSTOM_LOGIN_URL_MAPPED_TO_UNRESERVED_ADDRESS = 13
        CUSTOM_LOGIN_URL_HAS_NON_ROUTABLE_IP_ADDRESS = 14
        CUSTOM_LOGIN_URL_HAS_UNRESERVED_IP_ADDRESS = 15
        DUPLICATE_SCAN_NAME = 16
        INVALID_FIELD_VALUE = 17
        FAILED_TO_AUTHENTICATE_TO_TARGET = 18
        FINDING_TYPE_UNSPECIFIED = 19
        FORBIDDEN_TO_SCAN_COMPUTE = 20
        FORBIDDEN_UPDATE_TO_MANAGED_SCAN = 21
        MALFORMED_FILTER = 22
        MALFORMED_RESOURCE_NAME = 23
        PROJECT_INACTIVE = 24
        REQUIRED_FIELD = 25
        RESOURCE_NAME_INCONSISTENT = 26
        SCAN_ALREADY_RUNNING = 27
        SCAN_NOT_RUNNING = 28
        SEED_URL_DOES_NOT_BELONG_TO_CURRENT_PROJECT = 29
        SEED_URL_MALFORMED = 30
        SEED_URL_MAPPED_TO_NON_ROUTABLE_ADDRESS = 31
        SEED_URL_MAPPED_TO_UNRESERVED_ADDRESS = 32
        SEED_URL_HAS_NON_ROUTABLE_IP_ADDRESS = 33
        SEED_URL_HAS_UNRESERVED_IP_ADDRESS = 34
        SERVICE_ACCOUNT_NOT_CONFIGURED = 35
        TOO_MANY_SCANS = 36
        UNABLE_TO_RESOLVE_PROJECT_INFO = 37
        UNSUPPORTED_BLACKLIST_PATTERN_FORMAT = 38
        UNSUPPORTED_FILTER = 39
        UNSUPPORTED_FINDING_TYPE = 40
        UNSUPPORTED_URL_SCHEME = 41
        CLOUD_ASSET_INVENTORY_ASSET_NOT_FOUND = 42
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    fieldName = _messages.StringField(2)