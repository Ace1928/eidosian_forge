from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudRunStatus(_messages.Message):
    """CloudRunStatus defines the observed state of CloudRun

  Messages:
    AnnotationsValue: Annotations is additional Status fields for the Resource
      to save some additional State as well as convey more information to the
      user. This is roughly akin to Annotations on any k8s resource, just the
      reconciler conveying richer information outwards.

  Fields:
    annotations: Annotations is additional Status fields for the Resource to
      save some additional State as well as convey more information to the
      user. This is roughly akin to Annotations on any k8s resource, just the
      reconciler conveying richer information outwards.
    conditions: Conditions are the latest available observations of a
      resource's current state.
    eventingversion: The version of the installed release.
    istioversion: The version of the installed release.
    observedGeneration: ObservedGeneration is the 'Generation' of the Service
      that was last processed by the controller.
    servingversion: The version of the installed release.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations is additional Status fields for the Resource to save some
    additional State as well as convey more information to the user. This is
    roughly akin to Annotations on any k8s resource, just the reconciler
    conveying richer information outwards.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    conditions = _messages.MessageField('Condition', 2, repeated=True)
    eventingversion = _messages.StringField(3)
    istioversion = _messages.StringField(4)
    observedGeneration = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    servingversion = _messages.StringField(6)