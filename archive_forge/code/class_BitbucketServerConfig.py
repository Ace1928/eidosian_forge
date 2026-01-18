from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketServerConfig(_messages.Message):
    """BitbucketServerConfig represents the configuration for a Bitbucket
  Server.

  Fields:
    apiKey: Required. Immutable. API Key that will be attached to webhook.
      Once this field has been set, it cannot be changed. If you need to
      change it, please create another BitbucketServerConfig.
    connectedRepositories: Output only. Connected Bitbucket Server
      repositories for this config.
    createTime: Time when the config was created.
    hostUri: Required. Immutable. The URI of the Bitbucket Server host. Once
      this field has been set, it cannot be changed. If you need to change it,
      please create another BitbucketServerConfig.
    name: The resource name for the config.
    peeredNetwork: Optional. The network to be used when reaching out to the
      Bitbucket Server instance. The VPC network must be enabled for private
      service connection. This should be set if the Bitbucket Server instance
      is hosted on-premises and not reachable by public internet. If this
      field is left empty, no network peering will occur and calls to the
      Bitbucket Server instance will be made over the public internet. Must be
      in the format `projects/{project}/global/networks/{network}`, where
      {project} is a project number or id and {network} is the name of a VPC
      network in the project.
    peeredNetworkIpRange: Immutable. IP range within the peered network. This
      is specified in CIDR notation with a slash and the subnet prefix size.
      You can optionally specify an IP address before the subnet prefix value.
      e.g. `192.168.0.0/29` would specify an IP range starting at 192.168.0.0
      with a 29 bit prefix size. `/16` would specify a prefix size of 16 bits,
      with an automatically determined IP within the peered VPC. If
      unspecified, a value of `/24` will be used. The field only has an effect
      if peered_network is set.
    secrets: Required. Secret Manager secrets needed by the config.
    sslCa: Optional. SSL certificate to use for requests to Bitbucket Server.
      The format should be PEM format but the extension can be one of .pem,
      .cer, or .crt.
    username: Username of the account Cloud Build will use on Bitbucket
      Server.
    webhookKey: Output only. UUID included in webhook requests. The UUID is
      used to look up the corresponding config.
  """
    apiKey = _messages.StringField(1)
    connectedRepositories = _messages.MessageField('BitbucketServerRepositoryId', 2, repeated=True)
    createTime = _messages.StringField(3)
    hostUri = _messages.StringField(4)
    name = _messages.StringField(5)
    peeredNetwork = _messages.StringField(6)
    peeredNetworkIpRange = _messages.StringField(7)
    secrets = _messages.MessageField('BitbucketServerSecrets', 8)
    sslCa = _messages.StringField(9)
    username = _messages.StringField(10)
    webhookKey = _messages.StringField(11)