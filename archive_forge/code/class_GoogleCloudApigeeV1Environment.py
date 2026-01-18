from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Environment(_messages.Message):
    """A GoogleCloudApigeeV1Environment object.

  Enums:
    ApiProxyTypeValueValuesEnum: Optional. API Proxy type supported by the
      environment. The type can be set when creating the Environment and
      cannot be changed.
    DeploymentTypeValueValuesEnum: Optional. Deployment type supported by the
      environment. The deployment type can be set when creating the
      environment and cannot be changed. When you enable archive deployment,
      you will be **prevented from performing** a [subset of
      actions](/apigee/docs/api-platform/local-development/overview#prevented-
      actions) within the environment, including: * Managing the deployment of
      API proxy or shared flow revisions * Creating, updating, or deleting
      resource files * Creating, updating, or deleting target servers
    StateValueValuesEnum: Output only. State of the environment. Values other
      than ACTIVE means the resource is not ready to use.
    TypeValueValuesEnum: Optional. EnvironmentType selected for the
      environment.

  Fields:
    apiProxyType: Optional. API Proxy type supported by the environment. The
      type can be set when creating the Environment and cannot be changed.
    createdAt: Output only. Creation time of this environment as milliseconds
      since epoch.
    deploymentType: Optional. Deployment type supported by the environment.
      The deployment type can be set when creating the environment and cannot
      be changed. When you enable archive deployment, you will be **prevented
      from performing** a [subset of actions](/apigee/docs/api-platform/local-
      development/overview#prevented-actions) within the environment,
      including: * Managing the deployment of API proxy or shared flow
      revisions * Creating, updating, or deleting resource files * Creating,
      updating, or deleting target servers
    description: Optional. Description of the environment.
    displayName: Optional. Display name for this environment.
    forwardProxyUri: Optional. URI of the forward proxy to be applied to the
      runtime instances in this environment. Must be in the format of
      {scheme}://{hostname}:{port}. Note that the scheme must be one of "http"
      or "https", and the port must be supplied. To remove a forward proxy
      setting, update the field to an empty value. Note: At this time, PUT
      operations to add forwardProxyUri to an existing environment fail if the
      environment has nodeConfig set up. To successfully add the
      forwardProxyUri setting in this case, include the NodeConfig details
      with the request.
    hasAttachedFlowHooks: A boolean attribute.
    lastModifiedAt: Output only. Last modification time of this environment as
      milliseconds since epoch.
    name: Required. Name of the environment. Values must match the regular
      expression `^[.\\\\p{Alnum}-_]{1,255}$`
    nodeConfig: Optional. NodeConfig of the environment.
    properties: Optional. Key-value pairs that may be used for customizing the
      environment.
    state: Output only. State of the environment. Values other than ACTIVE
      means the resource is not ready to use.
    type: Optional. EnvironmentType selected for the environment.
  """

    class ApiProxyTypeValueValuesEnum(_messages.Enum):
        """Optional. API Proxy type supported by the environment. The type can be
    set when creating the Environment and cannot be changed.

    Values:
      API_PROXY_TYPE_UNSPECIFIED: API proxy type not specified.
      PROGRAMMABLE: Programmable API Proxies enable you to develop APIs with
        highly flexible behavior using bundled policy configuration and one or
        more programming languages to describe complex sequential and/or
        conditional flows of logic.
      CONFIGURABLE: Configurable API Proxies enable you to develop efficient
        APIs using simple configuration while complex execution control flow
        logic is handled by Apigee. This type only works with the ARCHIVE
        deployment type and cannot be combined with the PROXY deployment type.
    """
        API_PROXY_TYPE_UNSPECIFIED = 0
        PROGRAMMABLE = 1
        CONFIGURABLE = 2

    class DeploymentTypeValueValuesEnum(_messages.Enum):
        """Optional. Deployment type supported by the environment. The deployment
    type can be set when creating the environment and cannot be changed. When
    you enable archive deployment, you will be **prevented from performing** a
    [subset of actions](/apigee/docs/api-platform/local-
    development/overview#prevented-actions) within the environment, including:
    * Managing the deployment of API proxy or shared flow revisions *
    Creating, updating, or deleting resource files * Creating, updating, or
    deleting target servers

    Values:
      DEPLOYMENT_TYPE_UNSPECIFIED: Deployment type not specified.
      PROXY: Proxy deployment enables you to develop and deploy API proxies
        using Apigee on Google Cloud. This cannot currently be combined with
        the CONFIGURABLE API proxy type.
      ARCHIVE: Archive deployment enables you to develop API proxies locally
        then deploy an archive of your API proxy configuration to an
        environment in Apigee on Google Cloud. You will be prevented from
        performing a [subset of actions](/apigee/docs/api-platform/local-
        development/overview#prevented-actions) within the environment.
    """
        DEPLOYMENT_TYPE_UNSPECIFIED = 0
        PROXY = 1
        ARCHIVE = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the environment. Values other than ACTIVE means
    the resource is not ready to use.

    Values:
      STATE_UNSPECIFIED: Resource is in an unspecified state.
      CREATING: Resource is being created.
      ACTIVE: Resource is provisioned and ready to use.
      DELETING: The resource is being deleted.
      UPDATING: The resource is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        UPDATING = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. EnvironmentType selected for the environment.

    Values:
      ENVIRONMENT_TYPE_UNSPECIFIED: Environment type not specified.
      BASE: This is the default type. Base environment has limited capacity
        and capabilities and are usually used when you are getting started
        with Apigee or while experimenting. Refer to Apigee's public
        documentation for more details.
      INTERMEDIATE: Intermediate environment supports API management features
        and higher capacity than Base environment. Refer to Apigee's public
        documentation for more details.
      COMPREHENSIVE: Comprehensive environment supports advanced capabilites
        and even higher capacity than Intermediate environment. Refer to
        Apigee's public documentation for more details.
    """
        ENVIRONMENT_TYPE_UNSPECIFIED = 0
        BASE = 1
        INTERMEDIATE = 2
        COMPREHENSIVE = 3
    apiProxyType = _messages.EnumField('ApiProxyTypeValueValuesEnum', 1)
    createdAt = _messages.IntegerField(2)
    deploymentType = _messages.EnumField('DeploymentTypeValueValuesEnum', 3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    forwardProxyUri = _messages.StringField(6)
    hasAttachedFlowHooks = _messages.BooleanField(7)
    lastModifiedAt = _messages.IntegerField(8)
    name = _messages.StringField(9)
    nodeConfig = _messages.MessageField('GoogleCloudApigeeV1NodeConfig', 10)
    properties = _messages.MessageField('GoogleCloudApigeeV1Properties', 11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    type = _messages.EnumField('TypeValueValuesEnum', 13)