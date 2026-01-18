from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresSearchFeaturesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresSearchFeaturesRequest object.

  Fields:
    location: Required. The resource name of the Location to search Features.
      Format: `projects/{project}/locations/{location}`
    pageSize: The maximum number of Features to return. The service may return
      fewer than this value. If unspecified, at most 100 Features will be
      returned. The maximum value is 100; any value greater than 100 will be
      coerced to 100.
    pageToken: A page token, received from a previous
      FeaturestoreService.SearchFeatures call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      FeaturestoreService.SearchFeatures, except `page_size`, must match the
      call that provided the page token.
    query: Query string that is a conjunction of field-restricted queries
      and/or field-restricted filters. Field-restricted queries and filters
      can be combined using `AND` to form a conjunction. A field query is in
      the form FIELD:QUERY. This implicitly checks if QUERY exists as a
      substring within Feature's FIELD. The QUERY and the FIELD are converted
      to a sequence of words (i.e. tokens) for comparison. This is done by: *
      Removing leading/trailing whitespace and tokenizing the search value.
      Characters that are not one of alphanumeric `[a-zA-Z0-9]`, underscore
      `_`, or asterisk `*` are treated as delimiters for tokens. `*` is
      treated as a wildcard that matches characters within a token. * Ignoring
      case. * Prepending an asterisk to the first and appending an asterisk to
      the last token in QUERY. A QUERY must be either a singular token or a
      phrase. A phrase is one or multiple words enclosed in double quotation
      marks ("). With phrases, the order of the words is important. Words in
      the phrase must be matching in order and consecutively. Supported FIELDs
      for field-restricted queries: * `feature_id` * `description` *
      `entity_type_id` Examples: * `feature_id: foo` --> Matches a Feature
      with ID containing the substring `foo` (eg. `foo`, `foofeature`,
      `barfoo`). * `feature_id: foo*feature` --> Matches a Feature with ID
      containing the substring `foo*feature` (eg. `foobarfeature`). *
      `feature_id: foo AND description: bar` --> Matches a Feature with ID
      containing the substring `foo` and description containing the substring
      `bar`. Besides field queries, the following exact-match filters are
      supported. The exact-match filters do not support wildcards. Unlike
      field-restricted queries, exact-match filters are case-sensitive. *
      `feature_id`: Supports = comparisons. * `description`: Supports =
      comparisons. Multi-token filters should be enclosed in quotes. *
      `entity_type_id`: Supports = comparisons. * `value_type`: Supports = and
      != comparisons. * `labels`: Supports key-value equality as well as key
      presence. * `featurestore_id`: Supports = comparisons. Examples: *
      `description = "foo bar"` --> Any Feature with description exactly equal
      to `foo bar` * `value_type = DOUBLE` --> Features whose type is DOUBLE.
      * `labels.active = yes AND labels.env = prod` --> Features having both
      (active: yes) and (env: prod) labels. * `labels.env: *` --> Any Feature
      which has a label with `env` as the key.
  """
    location = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    query = _messages.StringField(4)