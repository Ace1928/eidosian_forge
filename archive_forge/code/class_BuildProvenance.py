from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildProvenance(_messages.Message):
    """Provenance of a build. Contains all information needed to verify the
  full details about the build from source to completion.

  Messages:
    BuildOptionsValue: Special options applied to this build. This is a catch-
      all field where build providers can enter any desired additional
      details.

  Fields:
    buildOptions: Special options applied to this build. This is a catch-all
      field where build providers can enter any desired additional details.
    builderVersion: Version string of the builder at the time this build was
      executed.
    builtArtifacts: Output of the build.
    commands: Commands requested by the build.
    createTime: Time at which the build was created.
    creator: E-mail address of the user who initiated this build. Note that
      this was the user's e-mail address at the time the build was initiated;
      this address may not represent the same end-user for all time.
    endTime: Time at which execution of the build was finished.
    id: Required. Unique identifier of the build.
    logsUri: URI where any logs for this provenance were written.
    projectId: ID of the project.
    sourceProvenance: Details of the Source input to the build.
    startTime: Time at which execution of the build was started.
    triggerId: Trigger identifier if the build was triggered automatically;
      empty if not.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BuildOptionsValue(_messages.Message):
        """Special options applied to this build. This is a catch-all field where
    build providers can enter any desired additional details.

    Messages:
      AdditionalProperty: An additional property for a BuildOptionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type BuildOptionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BuildOptionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    buildOptions = _messages.MessageField('BuildOptionsValue', 1)
    builderVersion = _messages.StringField(2)
    builtArtifacts = _messages.MessageField('Artifact', 3, repeated=True)
    commands = _messages.MessageField('Command', 4, repeated=True)
    createTime = _messages.StringField(5)
    creator = _messages.StringField(6)
    endTime = _messages.StringField(7)
    id = _messages.StringField(8)
    logsUri = _messages.StringField(9)
    projectId = _messages.StringField(10)
    sourceProvenance = _messages.MessageField('Source', 11)
    startTime = _messages.StringField(12)
    triggerId = _messages.StringField(13)