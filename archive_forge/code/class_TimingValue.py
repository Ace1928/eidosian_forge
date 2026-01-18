from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TimingValue(_messages.Message):
    """Output only. Stores timing information for phases of the build. Valid
    keys are: * BUILD: time to execute all build steps. * PUSH: time to push
    all artifacts including docker images and non docker artifacts. *
    FETCHSOURCE: time to fetch source. * SETUPBUILD: time to set up build. If
    the build does not specify source or images, these keys will not be
    included.

    Messages:
      AdditionalProperty: An additional property for a TimingValue object.

    Fields:
      additionalProperties: Additional properties of type TimingValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TimingValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleDevtoolsCloudbuildV1TimeSpan attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleDevtoolsCloudbuildV1TimeSpan', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)