from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryConfig(_messages.Message):
    """Configuration for discovery to scan resources for profile generation.
  Only one discovery configuration may exist per organization, folder, or
  project. The generated data profiles are retained according to the [data
  retention policy] (https://cloud.google.com/sensitive-data-
  protection/docs/data-profiles#retention).

  Enums:
    StatusValueValuesEnum: Required. A status for this configuration.

  Fields:
    actions: Actions to execute at the completion of scanning.
    createTime: Output only. The creation timestamp of a DiscoveryConfig.
    displayName: Display name (max 100 chars)
    errors: Output only. A stream of errors encountered when the config was
      activated. Repeated errors may result in the config automatically being
      paused. Output only field. Will return the last 100 errors. Whenever the
      config is modified this list will be cleared.
    inspectTemplates: Detection logic for profile generation. Not all template
      features are used by Discovery. FindingLimits, include_quote and
      exclude_info_types have no impact on Discovery. Multiple templates may
      be provided if there is data in multiple regions. At most one template
      must be specified per-region (including "global"). Each region is
      scanned using the applicable template. If no region-specific template is
      specified, but a "global" template is specified, it will be copied to
      that region and used instead. If no global or region-specific template
      is provided for a region with data, that region's data will not be
      scanned. For more information, see https://cloud.google.com/sensitive-
      data-protection/docs/data-profiles#data-residency.
    lastRunTime: Output only. The timestamp of the last time this config was
      executed.
    name: Unique resource name for the DiscoveryConfig, assigned by the
      service when the DiscoveryConfig is created, for example `projects/dlp-
      test-project/locations/global/discoveryConfigs/53234423`.
    orgConfig: Only set when the parent is an org.
    status: Required. A status for this configuration.
    targets: Target to match against for determining what to scan and how
      frequently.
    updateTime: Output only. The last update timestamp of a DiscoveryConfig.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Required. A status for this configuration.

    Values:
      STATUS_UNSPECIFIED: Unused
      RUNNING: The discovery config is currently active.
      PAUSED: The discovery config is paused temporarily.
    """
        STATUS_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
    actions = _messages.MessageField('GooglePrivacyDlpV2DataProfileAction', 1, repeated=True)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    errors = _messages.MessageField('GooglePrivacyDlpV2Error', 4, repeated=True)
    inspectTemplates = _messages.StringField(5, repeated=True)
    lastRunTime = _messages.StringField(6)
    name = _messages.StringField(7)
    orgConfig = _messages.MessageField('GooglePrivacyDlpV2OrgConfig', 8)
    status = _messages.EnumField('StatusValueValuesEnum', 9)
    targets = _messages.MessageField('GooglePrivacyDlpV2DiscoveryTarget', 10, repeated=True)
    updateTime = _messages.StringField(11)