from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1SourceContext(_messages.Message):
    """A SourceContext is a reference to a tree of files. A SourceContext
  together with a path point to a unique revision of a single file or
  directory.

  Messages:
    LabelsValue: Labels with user defined metadata.

  Fields:
    cloudRepo: A SourceContext referring to a revision in a Google Cloud
      Source Repo.
    gerrit: A SourceContext referring to a Gerrit project.
    git: A SourceContext referring to any third party Git repo (e.g., GitHub).
    labels: Labels with user defined metadata.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels with user defined metadata.

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
    cloudRepo = _messages.MessageField('GoogleDevtoolsContaineranalysisV1alpha1CloudRepoSourceContext', 1)
    gerrit = _messages.MessageField('GoogleDevtoolsContaineranalysisV1alpha1GerritSourceContext', 2)
    git = _messages.MessageField('GoogleDevtoolsContaineranalysisV1alpha1GitSourceContext', 3)
    labels = _messages.MessageField('LabelsValue', 4)