from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeAddonsConfig(_messages.Message):
    """RuntimeAddonsConfig defines the runtime configurations for add-ons in an
  environment.

  Fields:
    analyticsConfig: Runtime configuration for Analytics add-on.
    apiSecurityConfig: Runtime configuration for API Security add-on.
    name: Name of the addons config in the format:
      `organizations/{org}/environments/{env}/addonsConfig`
    revisionId: Revision number used by the runtime to detect config changes.
    uid: UID is to detect if config is recreated after deletion. The add-on
      config will only be deleted when the environment itself gets deleted,
      thus it will always be the same as the UID of EnvironmentConfig.
  """
    analyticsConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeAnalyticsConfig', 1)
    apiSecurityConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeApiSecurityConfig', 2)
    name = _messages.StringField(3)
    revisionId = _messages.StringField(4)
    uid = _messages.StringField(5)