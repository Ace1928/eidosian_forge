from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DebugMask(_messages.Message):
    """A GoogleCloudApigeeV1DebugMask object.

  Messages:
    NamespacesValue: Map of namespaces to URIs.

  Fields:
    faultJSONPaths: List of JSON paths that specify the JSON elements to be
      filtered from JSON payloads in error flows.
    faultXPaths: List of XPaths that specify the XML elements to be filtered
      from XML payloads in error flows.
    name: Name of the debug mask.
    namespaces: Map of namespaces to URIs.
    requestJSONPaths: List of JSON paths that specify the JSON elements to be
      filtered from JSON request message payloads.
    requestXPaths: List of XPaths that specify the XML elements to be filtered
      from XML request message payloads.
    responseJSONPaths: List of JSON paths that specify the JSON elements to be
      filtered from JSON response message payloads.
    responseXPaths: List of XPaths that specify the XML elements to be
      filtered from XML response message payloads.
    variables: List of variables that should be masked from the debug output.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NamespacesValue(_messages.Message):
        """Map of namespaces to URIs.

    Messages:
      AdditionalProperty: An additional property for a NamespacesValue object.

    Fields:
      additionalProperties: Additional properties of type NamespacesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NamespacesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    faultJSONPaths = _messages.StringField(1, repeated=True)
    faultXPaths = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    namespaces = _messages.MessageField('NamespacesValue', 4)
    requestJSONPaths = _messages.StringField(5, repeated=True)
    requestXPaths = _messages.StringField(6, repeated=True)
    responseJSONPaths = _messages.StringField(7, repeated=True)
    responseXPaths = _messages.StringField(8, repeated=True)
    variables = _messages.StringField(9, repeated=True)