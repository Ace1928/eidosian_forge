from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlinkJob(_messages.Message):
    """A Dataproc job for running Apache Flink applications on YARN.

  Messages:
    PropertiesValue: Optional. A mapping of property names to values, used to
      configure Flink. Properties that conflict with values set by the
      Dataproc API might beoverwritten. Can include properties set
      in/etc/flink/conf/flink-defaults.conf and classes in user code.

  Fields:
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as --conf, that can be set as job properties, since a
      collision might occur that causes an incorrect job submission.
    jarFileUris: Optional. HCFS URIs of jar files to add to the CLASSPATHs of
      the Flink driver and tasks.
    loggingConfig: Optional. The runtime log config for job execution.
    mainClass: The name of the driver's main class. The jar file that contains
      the class must be in the default CLASSPATH or specified in jarFileUris.
    mainJarFileUri: The HCFS URI of the jar file that contains the main class.
    properties: Optional. A mapping of property names to values, used to
      configure Flink. Properties that conflict with values set by the
      Dataproc API might beoverwritten. Can include properties set
      in/etc/flink/conf/flink-defaults.conf and classes in user code.
    savepointUri: Optional. HCFS URI of the savepoint, which contains the last
      saved progress for starting the current job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values, used to configure
    Flink. Properties that conflict with values set by the Dataproc API might
    beoverwritten. Can include properties set in/etc/flink/conf/flink-
    defaults.conf and classes in user code.

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
    args = _messages.StringField(1, repeated=True)
    jarFileUris = _messages.StringField(2, repeated=True)
    loggingConfig = _messages.MessageField('LoggingConfig', 3)
    mainClass = _messages.StringField(4)
    mainJarFileUri = _messages.StringField(5)
    properties = _messages.MessageField('PropertiesValue', 6)
    savepointUri = _messages.StringField(7)