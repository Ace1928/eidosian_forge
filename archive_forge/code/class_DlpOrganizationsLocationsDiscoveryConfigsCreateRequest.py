from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpOrganizationsLocationsDiscoveryConfigsCreateRequest(_messages.Message):
    """A DlpOrganizationsLocationsDiscoveryConfigsCreateRequest object.

  Fields:
    googlePrivacyDlpV2CreateDiscoveryConfigRequest: A
      GooglePrivacyDlpV2CreateDiscoveryConfigRequest resource to be passed as
      the request body.
    parent: Required. Parent resource name. The format of this value is as
      follows: `projects/`PROJECT_ID`/locations/`LOCATION_ID The following
      example `parent` string specifies a parent project with the identifier
      `example-project`, and specifies the `europe-west3` location for
      processing data: parent=projects/example-project/locations/europe-west3
  """
    googlePrivacyDlpV2CreateDiscoveryConfigRequest = _messages.MessageField('GooglePrivacyDlpV2CreateDiscoveryConfigRequest', 1)
    parent = _messages.StringField(2, required=True)