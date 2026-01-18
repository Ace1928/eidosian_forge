from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogSink(_messages.Message):
    """Describes a sink used to export log entries to one of the following
  destinations: a Cloud Logging log bucket, a Cloud Storage bucket, a BigQuery
  dataset, a Pub/Sub topic, a Cloud project.A logs filter controls which log
  entries are exported. The sink must be created within a project,
  organization, billing account, or folder.

  Enums:
    OutputVersionFormatValueValuesEnum: Deprecated. This field is unused.

  Fields:
    bigqueryOptions: Optional. Options that affect sinks exporting data to
      BigQuery.
    createTime: Output only. The creation timestamp of the sink.This field may
      not be present for older sinks.
    description: Optional. A description of this sink.The maximum length of
      the description is 8000 characters.
    destination: Required. The export destination:
      "storage.googleapis.com/[GCS_BUCKET]"
      "bigquery.googleapis.com/projects/[PROJECT_ID]/datasets/[DATASET]"
      "pubsub.googleapis.com/projects/[PROJECT_ID]/topics/[TOPIC_ID]"
      "logging.googleapis.com/projects/[PROJECT_ID]" "logging.googleapis.com/p
      rojects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" The
      sink's writer_identity, set when the sink is created, must have
      permission to write to the destination or else the log entries are not
      exported. For more information, see Exporting Logs with Sinks
      (https://cloud.google.com/logging/docs/api/tasks/exporting-logs).
    disabled: Optional. If set to true, then this sink is disabled and it does
      not export any log entries.
    exclusions: Optional. Log entries that match any of these exclusion
      filters will not be exported.If a log entry is matched by both filter
      and one of exclusion_filters it will not be exported.
    filter: Optional. An advanced logs filter
      (https://cloud.google.com/logging/docs/view/advanced-queries). The only
      exported log entries are those that are in the resource owning the sink
      and that match the filter.For
      example:logName="projects/[PROJECT_ID]/logs/[LOG_ID]" AND
      severity>=ERROR
    includeChildren: Optional. This field applies only to sinks owned by
      organizations and folders. If the field is false, the default, only the
      logs owned by the sink's parent resource are available for export. If
      the field is true, then log entries from all the projects, folders, and
      billing accounts contained in the sink's parent resource are also
      available for export. Whether a particular log entry from the children
      is exported depends on the sink's filter expression.For example, if this
      field is true, then the filter resource.type=gce_instance would export
      all Compute Engine VM instance log entries from all projects in the
      sink's parent.To only export entries from certain child projects, filter
      on the project part of the log name:logName:("projects/test-project1/"
      OR "projects/test-project2/") AND resource.type=gce_instance
    interceptChildren: Optional. This field applies only to sinks owned by
      organizations and folders.When the value of 'intercept_children' is
      true, the following restrictions apply: The sink must have the
      include_children flag set to true. The sink destination must be a Cloud
      project.Also, the following behaviors apply: Any logs matched by the
      sink won't be included by non-_Required sinks owned by child resources.
      The sink appears in the results of a ListSinks call from a child
      resource if the value of the filter field in its request is either
      'in_scope("ALL")' or 'in_scope("ANCESTOR")'.
    name: Output only. The client-assigned sink identifier, unique within the
      project.For example: "my-syslog-errors-to-pubsub".Sink identifiers are
      limited to 100 characters and can include only the following characters:
      upper and lower-case alphanumeric characters, underscores, hyphens,
      periods.First character has to be alphanumeric.
    outputVersionFormat: Deprecated. This field is unused.
    resourceName: Output only. The resource name of the sink.
      "projects/[PROJECT_ID]/sinks/[SINK_NAME]
      "organizations/[ORGANIZATION_ID]/sinks/[SINK_NAME]
      "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_NAME]
      "folders/[FOLDER_ID]/sinks/[SINK_NAME] For example:
      projects/my_project/sinks/SINK_NAME
    updateTime: Output only. The last update timestamp of the sink.This field
      may not be present for older sinks.
    writerIdentity: Output only. An IAM identity-a service account or group-
      under which Cloud Logging writes the exported log entries to the sink's
      destination. This field is either set by specifying
      custom_writer_identity or set automatically by sinks.create and
      sinks.update based on the value of unique_writer_identity in those
      methods.Until you grant this identity write-access to the destination,
      log entry exports from this sink will fail. For more information, see
      Granting Access for a Resource
      (https://cloud.google.com/iam/docs/granting-roles-to-service-
      accounts#granting_access_to_a_service_account_for_a_resource). Consult
      the destination service's documentation to determine the appropriate IAM
      roles to assign to the identity.Sinks that have a destination that is a
      log bucket in the same project as the sink cannot have a writer_identity
      and no additional permissions are required.
  """

    class OutputVersionFormatValueValuesEnum(_messages.Enum):
        """Deprecated. This field is unused.

    Values:
      VERSION_FORMAT_UNSPECIFIED: An unspecified format version that will
        default to V2.
      V2: LogEntry version 2 format.
      V1: LogEntry version 1 format.
    """
        VERSION_FORMAT_UNSPECIFIED = 0
        V2 = 1
        V1 = 2
    bigqueryOptions = _messages.MessageField('BigQueryOptions', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    destination = _messages.StringField(4)
    disabled = _messages.BooleanField(5)
    exclusions = _messages.MessageField('LogExclusion', 6, repeated=True)
    filter = _messages.StringField(7)
    includeChildren = _messages.BooleanField(8)
    interceptChildren = _messages.BooleanField(9)
    name = _messages.StringField(10)
    outputVersionFormat = _messages.EnumField('OutputVersionFormatValueValuesEnum', 11)
    resourceName = _messages.StringField(12)
    updateTime = _messages.StringField(13)
    writerIdentity = _messages.StringField(14)