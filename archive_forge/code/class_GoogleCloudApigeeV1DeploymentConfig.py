from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentConfig(_messages.Message):
    """NEXT ID: 11

  Messages:
    AttributesValue: Additional key-value metadata for the deployment.
    EndpointsValue: A mapping from basepaths to proxy endpoint names in this
      proxy. Not populated for shared flows.

  Fields:
    attributes: Additional key-value metadata for the deployment.
    basePath: Base path where the application will be hosted. Defaults to "/".
    deploymentGroups: The list of deployment groups in which this proxy should
      be deployed. Not currently populated for shared flows.
    endpoints: A mapping from basepaths to proxy endpoint names in this proxy.
      Not populated for shared flows.
    location: Location of the API proxy bundle as a URI.
    name: Name of the API or shared flow revision to be deployed in the
      following format: `organizations/{org}/apis/{api}/revisions/{rev}` or
      `organizations/{org}/sharedflows/{sharedflow}/revisions/{rev}`
    proxyUid: Unique ID of the API proxy revision.
    serviceAccount: The service account identity associated with this
      deployment. If non-empty, will be in the following format:
      `projects/-/serviceAccounts/{account_email}`
    uid: Unique ID. The ID will only change if the deployment is deleted and
      recreated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Additional key-value metadata for the deployment.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EndpointsValue(_messages.Message):
        """A mapping from basepaths to proxy endpoint names in this proxy. Not
    populated for shared flows.

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
    attributes = _messages.MessageField('AttributesValue', 1)
    basePath = _messages.StringField(2)
    deploymentGroups = _messages.StringField(3, repeated=True)
    endpoints = _messages.MessageField('EndpointsValue', 4)
    location = _messages.StringField(5)
    name = _messages.StringField(6)
    proxyUid = _messages.StringField(7)
    serviceAccount = _messages.StringField(8)
    uid = _messages.StringField(9)