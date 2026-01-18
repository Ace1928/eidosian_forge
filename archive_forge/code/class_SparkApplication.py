from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkApplication(_messages.Message):
    """Represents the SparkApplication resource.

  Enums:
    RequestedStateValueValuesEnum: Optional. The intended state to which the
      application is reconciling.
    StateValueValuesEnum: Output only. The current state.

  Messages:
    AnnotationsValue: Optional. The annotations to associate with this
      application. Annotations may be used to store client information, but
      are not used by the server.
    LabelsValue: Optional. The labels to associate with this application.
      Labels may be used for filtering and billing tracking.
    PropertiesValue: Optional. application-specific properties.

  Fields:
    annotations: Optional. The annotations to associate with this application.
      Annotations may be used to store client information, but are not used by
      the server.
    applicationEnvironment: Optional. An ApplicationEnvironment from which to
      inherit configuration properties.
    createTime: Output only. The timestamp when the resource was created.
    displayName: Optional. User-provided human-readable name to be used in
      user interfaces.
    labels: Optional. The labels to associate with this application. Labels
      may be used for filtering and billing tracking.
    monitoringEndpoint: Output only. URL for a monitoring UI for this
      application (for eventual Spark PHS/UI support) Out of scope for private
      GA
    name: Identifier. Fields 1-6 should exist for all declarative friendly
      resources per aip.dev/148 The name of the application. Format: projects/
      {project}/locations/{location}/serviceInstances/{service_instance}/spark
      Applications/{application}
    namespace: Optional. The Kubernetes namespace in which to create the
      application. This namespace must already exist on the cluster.
    outputUri: Output only. An HCFS URI pointing to the location of stdout and
      stdout of the application Mainly useful for Pantheon and gcloud Not in
      scope for private GA
    properties: Optional. application-specific properties.
    pysparkApplicationConfig: PySpark application config.
    reconciling: Output only. Whether the application is currently
      reconciling. True if the current state of the resource does not match
      the intended state, and the system is working to reconcile them, whether
      or not the change was user initiated. Required by
      aip.dev/128#reconciliation
    requestedState: Optional. The intended state to which the application is
      reconciling.
    sparkApplicationConfig: Spark application config.
    sparkRApplicationConfig: SparkR application config.
    sparkSqlApplicationConfig: SparkSql application config.
    state: Output only. The current state.
    stateMessage: Output only. A message explaining the current state.
    uid: Output only. System generated unique identifier for this application,
      formatted as UUID4.
    updateTime: Output only. The timestamp when the resource was most recently
      updated.
    version: Optional. The Dataproc version of this application.
  """

    class RequestedStateValueValuesEnum(_messages.Enum):
        """Optional. The intended state to which the application is reconciling.

    Values:
      STATE_UNSPECIFIED: The application state is unknown.
      PENDING: The application is setting up and has not yet begun to execute
      RUNNING: The application is running.
      CANCELLING: The application is being cancelled.
      CANCELLED: The application was successfully cancelled
      SUCCEEDED: The application completed successfully.
      FAILED: The application exited with an error.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        CANCELLING = 3
        CANCELLED = 4
        SUCCEEDED = 5
        FAILED = 6

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state.

    Values:
      STATE_UNSPECIFIED: The application state is unknown.
      PENDING: The application is setting up and has not yet begun to execute
      RUNNING: The application is running.
      CANCELLING: The application is being cancelled.
      CANCELLED: The application was successfully cancelled
      SUCCEEDED: The application completed successfully.
      FAILED: The application exited with an error.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        CANCELLING = 3
        CANCELLED = 4
        SUCCEEDED = 5
        FAILED = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. The annotations to associate with this application.
    Annotations may be used to store client information, but are not used by
    the server.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this application. Labels may be
    used for filtering and billing tracking.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. application-specific properties.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    applicationEnvironment = _messages.StringField(2)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    monitoringEndpoint = _messages.StringField(6)
    name = _messages.StringField(7)
    namespace = _messages.StringField(8)
    outputUri = _messages.StringField(9)
    properties = _messages.MessageField('PropertiesValue', 10)
    pysparkApplicationConfig = _messages.MessageField('PySparkApplicationConfig', 11)
    reconciling = _messages.BooleanField(12)
    requestedState = _messages.EnumField('RequestedStateValueValuesEnum', 13)
    sparkApplicationConfig = _messages.MessageField('SparkApplicationConfig', 14)
    sparkRApplicationConfig = _messages.MessageField('SparkRApplicationConfig', 15)
    sparkSqlApplicationConfig = _messages.MessageField('SparkSqlApplicationConfig', 16)
    state = _messages.EnumField('StateValueValuesEnum', 17)
    stateMessage = _messages.StringField(18)
    uid = _messages.StringField(19)
    updateTime = _messages.StringField(20)
    version = _messages.StringField(21)