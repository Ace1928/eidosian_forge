from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkStatistics(_messages.Message):
    """Statistics for a BigSpark query. Populated as part of JobStatistics2

  Messages:
    EndpointsValue: Output only. Endpoints returned from Dataproc. Key list: -
      history_server_endpoint: A link to Spark job UI.

  Fields:
    endpoints: Output only. Endpoints returned from Dataproc. Key list: -
      history_server_endpoint: A link to Spark job UI.
    gcsStagingBucket: Output only. The Google Cloud Storage bucket that is
      used as the default file system by the Spark application. This field is
      only filled when the Spark procedure uses the invoker security mode. The
      `gcsStagingBucket` bucket is inferred from the
      `@@spark_proc_properties.staging_bucket` system variable (if it is
      provided). Otherwise, BigQuery creates a default staging bucket for the
      job and returns the bucket name in this field. Example: *
      `gs://[bucket_name]`
    kmsKeyName: Output only. The Cloud KMS encryption key that is used to
      protect the resources created by the Spark job. If the Spark procedure
      uses the invoker security mode, the Cloud KMS encryption key is either
      inferred from the provided system variable,
      `@@spark_proc_properties.kms_key_name`, or the default key of the
      BigQuery job's project (if the CMEK organization policy is enforced).
      Otherwise, the Cloud KMS key is either inferred from the Spark
      connection associated with the procedure (if it is provided), or from
      the default key of the Spark connection's project if the CMEK
      organization policy is enforced. Example: * `projects/[kms_project_id]/l
      ocations/[region]/keyRings/[key_region]/cryptoKeys/[key]`
    loggingInfo: Output only. Logging info is used to generate a link to Cloud
      Logging.
    sparkJobId: Output only. Spark job ID if a Spark job is created
      successfully.
    sparkJobLocation: Output only. Location where the Spark job is executed. A
      location is selected by BigQueury for jobs configured to run in a multi-
      region.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EndpointsValue(_messages.Message):
        """Output only. Endpoints returned from Dataproc. Key list: -
    history_server_endpoint: A link to Spark job UI.

    Messages:
      AdditionalProperty: An additional property for a EndpointsValue object.

    Fields:
      additionalProperties: Additional properties of type EndpointsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EndpointsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    endpoints = _messages.MessageField('EndpointsValue', 1)
    gcsStagingBucket = _messages.StringField(2)
    kmsKeyName = _messages.StringField(3)
    loggingInfo = _messages.MessageField('SparkLoggingInfo', 4)
    sparkJobId = _messages.StringField(5)
    sparkJobLocation = _messages.StringField(6)