from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UptimeCheckConfig(_messages.Message):
    """This message configures which resources and services to monitor for
  availability.

  Enums:
    CheckerTypeValueValuesEnum: The type of checkers to use to execute the
      Uptime check.
    SelectedRegionsValueListEntryValuesEnum:

  Messages:
    UserLabelsValue: User-supplied key/value data to be used for organizing
      and identifying the UptimeCheckConfig objects.The field can contain up
      to 64 entries. Each key and value is limited to 63 Unicode characters or
      128 bytes, whichever is smaller. Labels and values can contain only
      lowercase letters, numerals, underscores, and dashes. Keys must begin
      with a letter.

  Fields:
    checkerType: The type of checkers to use to execute the Uptime check.
    contentMatchers: The content that is expected to appear in the data
      returned by the target server against which the check is run. Currently,
      only the first entry in the content_matchers list is supported, and
      additional entries will be ignored. This field is optional and should
      only be specified if a content match is required as part of the/ Uptime
      check.
    displayName: A human-friendly name for the Uptime check configuration. The
      display name should be unique within a Cloud Monitoring Workspace in
      order to make it easier to identify; however, uniqueness is not
      enforced. Required.
    httpCheck: Contains information needed to make an HTTP or HTTPS check.
    internalCheckers: The internal checkers that this check will egress from.
      If is_internal is true and this list is empty, the check will egress
      from all the InternalCheckers configured for the project that owns this
      UptimeCheckConfig.
    isInternal: If this is true, then checks are made only from the
      'internal_checkers'. If it is false, then checks are made only from the
      'selected_regions'. It is an error to provide 'selected_regions' when
      is_internal is true, or to provide 'internal_checkers' when is_internal
      is false.
    monitoredResource: The monitored resource
      (https://cloud.google.com/monitoring/api/resources) associated with the
      configuration. The following monitored resource types are valid for this
      field: uptime_url, gce_instance, gae_app, aws_ec2_instance,
      aws_elb_load_balancer k8s_service servicedirectory_service
      cloud_run_revision
    name: Identifier. A unique resource name for this Uptime check
      configuration. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]
      [PROJECT_ID_OR_NUMBER] is the Workspace host project associated with the
      Uptime check.This field should be omitted when creating the Uptime check
      configuration; on create, the resource name is assigned by the server
      and included in the response.
    period: How often, in seconds, the Uptime check is performed. Currently,
      the only supported values are 60s (1 minute), 300s (5 minutes), 600s (10
      minutes), and 900s (15 minutes). Optional, defaults to 60s.
    resourceGroup: The group resource associated with the configuration.
    selectedRegions: The list of regions from which the check will be run.
      Some regions contain one location, and others contain more than one. If
      this field is specified, enough regions must be provided to include a
      minimum of 3 locations. Not specifying this field will result in Uptime
      checks running from all available regions.
    syntheticMonitor: Specifies a Synthetic Monitor to invoke.
    tcpCheck: Contains information needed to make a TCP check.
    timeout: The maximum amount of time to wait for the request to complete
      (must be between 1 and 60 seconds). Required.
    userLabels: User-supplied key/value data to be used for organizing and
      identifying the UptimeCheckConfig objects.The field can contain up to 64
      entries. Each key and value is limited to 63 Unicode characters or 128
      bytes, whichever is smaller. Labels and values can contain only
      lowercase letters, numerals, underscores, and dashes. Keys must begin
      with a letter.
  """

    class CheckerTypeValueValuesEnum(_messages.Enum):
        """The type of checkers to use to execute the Uptime check.

    Values:
      CHECKER_TYPE_UNSPECIFIED: The default checker type. Currently converted
        to STATIC_IP_CHECKERS on creation, the default conversion behavior may
        change in the future.
      STATIC_IP_CHECKERS: STATIC_IP_CHECKERS are used for uptime checks that
        perform egress across the public internet. STATIC_IP_CHECKERS use the
        static IP addresses returned by ListUptimeCheckIps.
      VPC_CHECKERS: VPC_CHECKERS are used for uptime checks that perform
        egress using Service Directory and private network access. When using
        VPC_CHECKERS, the monitored resource type must be
        servicedirectory_service.
    """
        CHECKER_TYPE_UNSPECIFIED = 0
        STATIC_IP_CHECKERS = 1
        VPC_CHECKERS = 2

    class SelectedRegionsValueListEntryValuesEnum(_messages.Enum):
        """SelectedRegionsValueListEntryValuesEnum enum type.

    Values:
      REGION_UNSPECIFIED: Default value if no region is specified. Will result
        in Uptime checks running from all regions.
      USA: Allows checks to run from locations within the United States of
        America.
      EUROPE: Allows checks to run from locations within the continent of
        Europe.
      SOUTH_AMERICA: Allows checks to run from locations within the continent
        of South America.
      ASIA_PACIFIC: Allows checks to run from locations within the Asia
        Pacific area (ex: Singapore).
      USA_OREGON: Allows checks to run from locations within the western
        United States of America
      USA_IOWA: Allows checks to run from locations within the central United
        States of America
      USA_VIRGINIA: Allows checks to run from locations within the eastern
        United States of America
    """
        REGION_UNSPECIFIED = 0
        USA = 1
        EUROPE = 2
        SOUTH_AMERICA = 3
        ASIA_PACIFIC = 4
        USA_OREGON = 5
        USA_IOWA = 6
        USA_VIRGINIA = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """User-supplied key/value data to be used for organizing and identifying
    the UptimeCheckConfig objects.The field can contain up to 64 entries. Each
    key and value is limited to 63 Unicode characters or 128 bytes, whichever
    is smaller. Labels and values can contain only lowercase letters,
    numerals, underscores, and dashes. Keys must begin with a letter.

    Messages:
      AdditionalProperty: An additional property for a UserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type UserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    checkerType = _messages.EnumField('CheckerTypeValueValuesEnum', 1)
    contentMatchers = _messages.MessageField('ContentMatcher', 2, repeated=True)
    displayName = _messages.StringField(3)
    httpCheck = _messages.MessageField('HttpCheck', 4)
    internalCheckers = _messages.MessageField('InternalChecker', 5, repeated=True)
    isInternal = _messages.BooleanField(6)
    monitoredResource = _messages.MessageField('MonitoredResource', 7)
    name = _messages.StringField(8)
    period = _messages.StringField(9)
    resourceGroup = _messages.MessageField('ResourceGroup', 10)
    selectedRegions = _messages.EnumField('SelectedRegionsValueListEntryValuesEnum', 11, repeated=True)
    syntheticMonitor = _messages.MessageField('SyntheticMonitorTarget', 12)
    tcpCheck = _messages.MessageField('TcpCheck', 13)
    timeout = _messages.StringField(14)
    userLabels = _messages.MessageField('UserLabelsValue', 15)