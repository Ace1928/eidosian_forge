from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDeidentifyTemplatesCreateRequest(_messages.Message):
    """A DlpProjectsLocationsDeidentifyTemplatesCreateRequest object.

  Fields:
    googlePrivacyDlpV2CreateDeidentifyTemplateRequest: A
      GooglePrivacyDlpV2CreateDeidentifyTemplateRequest resource to be passed
      as the request body.
    parent: Required. Parent resource name. The format of this value varies
      depending on the scope of the request (project or organization) and
      whether you have [specified a processing
      location](https://cloud.google.com/sensitive-data-
      protection/docs/specifying-location): + Projects scope, location
      specified: `projects/`PROJECT_ID`/locations/`LOCATION_ID + Projects
      scope, no location specified (defaults to global): `projects/`PROJECT_ID
      + Organizations scope, location specified:
      `organizations/`ORG_ID`/locations/`LOCATION_ID + Organizations scope, no
      location specified (defaults to global): `organizations/`ORG_ID The
      following example `parent` string specifies a parent project with the
      identifier `example-project`, and specifies the `europe-west3` location
      for processing data: parent=projects/example-project/locations/europe-
      west3
  """
    googlePrivacyDlpV2CreateDeidentifyTemplateRequest = _messages.MessageField('GooglePrivacyDlpV2CreateDeidentifyTemplateRequest', 1)
    parent = _messages.StringField(2, required=True)