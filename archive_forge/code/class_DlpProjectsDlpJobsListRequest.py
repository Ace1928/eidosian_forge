from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsDlpJobsListRequest(_messages.Message):
    """A DlpProjectsDlpJobsListRequest object.

  Enums:
    TypeValueValuesEnum: The type of job. Defaults to `DlpJobType.INSPECT`

  Fields:
    filter: Allows filtering. Supported syntax: * Filter expressions are made
      up of one or more restrictions. * Restrictions can be combined by `AND`
      or `OR` logical operators. A sequence of restrictions implicitly uses
      `AND`. * A restriction has the form of `{field} {operator} {value}`. *
      Supported fields/values for inspect jobs: - `state` -
      PENDING|RUNNING|CANCELED|FINISHED|FAILED - `inspected_storage` -
      DATASTORE|CLOUD_STORAGE|BIGQUERY - `trigger_name` - The name of the
      trigger that created the job. - 'end_time` - Corresponds to the time the
      job finished. - 'start_time` - Corresponds to the time the job finished.
      * Supported fields for risk analysis jobs: - `state` -
      RUNNING|CANCELED|FINISHED|FAILED - 'end_time` - Corresponds to the time
      the job finished. - 'start_time` - Corresponds to the time the job
      finished. * The operator must be `=` or `!=`. Examples: *
      inspected_storage = cloud_storage AND state = done * inspected_storage =
      cloud_storage OR inspected_storage = bigquery * inspected_storage =
      cloud_storage AND (state = done OR state = canceled) * end_time >
      \\"2017-12-12T00:00:00+00:00\\" The length of this field should be no more
      than 500 characters.
    locationId: Deprecated. This field has no effect.
    orderBy: Comma separated list of fields to order by, followed by `asc` or
      `desc` postfix. This list is case insensitive. The default sorting order
      is ascending. Redundant space characters are insignificant. Example:
      `name asc, end_time asc, create_time desc` Supported fields are: -
      `create_time`: corresponds to the time the job was created. -
      `end_time`: corresponds to the time the job ended. - `name`: corresponds
      to the job's name. - `state`: corresponds to `state`
    pageSize: The standard list page size.
    pageToken: The standard list page token.
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
    type: The type of job. Defaults to `DlpJobType.INSPECT`
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of job. Defaults to `DlpJobType.INSPECT`

    Values:
      DLP_JOB_TYPE_UNSPECIFIED: Defaults to INSPECT_JOB.
      INSPECT_JOB: The job inspected Google Cloud for sensitive data.
      RISK_ANALYSIS_JOB: The job executed a Risk Analysis computation.
    """
        DLP_JOB_TYPE_UNSPECIFIED = 0
        INSPECT_JOB = 1
        RISK_ANALYSIS_JOB = 2
    filter = _messages.StringField(1)
    locationId = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)
    type = _messages.EnumField('TypeValueValuesEnum', 7)