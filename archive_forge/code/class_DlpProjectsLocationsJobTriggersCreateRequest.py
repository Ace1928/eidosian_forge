from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsJobTriggersCreateRequest(_messages.Message):
    """A DlpProjectsLocationsJobTriggersCreateRequest object.

  Fields:
    googlePrivacyDlpV2CreateJobTriggerRequest: A
      GooglePrivacyDlpV2CreateJobTriggerRequest resource to be passed as the
      request body.
    parent: Required. Parent resource name. The format of this value varies
      depending on whether you have [specified a processing
      location](https://cloud.google.com/sensitive-data-
      protection/docs/specifying-location): + Projects scope, location
      specified: `projects/`PROJECT_ID`/locations/`LOCATION_ID + Projects
      scope, no location specified (defaults to global): `projects/`PROJECT_ID
      The following example `parent` string specifies a parent project with
      the identifier `example-project`, and specifies the `europe-west3`
      location for processing data: parent=projects/example-
      project/locations/europe-west3
  """
    googlePrivacyDlpV2CreateJobTriggerRequest = _messages.MessageField('GooglePrivacyDlpV2CreateJobTriggerRequest', 1)
    parent = _messages.StringField(2, required=True)