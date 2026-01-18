from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDiscoveryConfigsPatchRequest(_messages.Message):
    """A DlpProjectsLocationsDiscoveryConfigsPatchRequest object.

  Fields:
    googlePrivacyDlpV2UpdateDiscoveryConfigRequest: A
      GooglePrivacyDlpV2UpdateDiscoveryConfigRequest resource to be passed as
      the request body.
    name: Required. Resource name of the project and the configuration, for
      example `projects/dlp-test-project/discoveryConfigs/53234423`.
  """
    googlePrivacyDlpV2UpdateDiscoveryConfigRequest = _messages.MessageField('GooglePrivacyDlpV2UpdateDiscoveryConfigRequest', 1)
    name = _messages.StringField(2, required=True)