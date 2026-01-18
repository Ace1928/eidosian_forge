from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ScopeSpecsValue(_messages.Message):
    """Optional. Scope-specific configuration for this Feature. If this
    Feature does not support any per-Scope configuration, this field may be
    unused. The keys indicate which Scope the configuration is for, in the
    form: `projects/{p}/locations/global/scopes/{s}` Where {p} is the project,
    {s} is a valid Scope in this project. {p} WILL match the Feature's
    project. {p} will always be returned as the project number, but the
    project ID is also accepted during input. If the same Scope is specified
    in the map twice (using the project ID form, and the project number form),
    exactly ONE of the entries will be saved, with no guarantees as to which.
    For this reason, it is recommended the same format be used for all entries
    when mutating a Feature.

    Messages:
      AdditionalProperty: An additional property for a ScopeSpecsValue object.

    Fields:
      additionalProperties: Additional properties of type ScopeSpecsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ScopeSpecsValue object.

      Fields:
        key: Name of the additional property.
        value: A ScopeFeatureSpec attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ScopeFeatureSpec', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)