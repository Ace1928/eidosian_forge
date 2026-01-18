from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubEnterpriseConfig(_messages.Message):
    """GitHubEnterpriseConfig represents a configuration for a GitHub
  Enterprise server.

  Fields:
    appConfigJson: Cloud Storage location of the encrypted GitHub App config
      information.
    appId: Required. The GitHub app id of the Cloud Build app on the GitHub
      Enterprise server.
    createTime: Output only. Time when the installation was associated with
      the project.
    displayName: Name to display for this config.
    hostUrl: The URL of the github enterprise host the configuration is for.
    name: Optional. The full resource name for the GitHubEnterpriseConfig For
      example: "projects/{$project_id}/locations/{$location_id}/githubEnterpri
      seConfigs/{$config_id}"
    peeredNetwork: Optional. The network to be used when reaching out to the
      GitHub Enterprise server. The VPC network must be enabled for private
      service connection. This should be set if the GitHub Enterprise server
      is hosted on-premises and not reachable by public internet. If this
      field is left empty, no network peering will occur and calls to the
      GitHub Enterprise server will be made over the public internet. Must be
      in the format `projects/{project}/global/networks/{network}`, where
      {project} is a project number or id and {network} is the name of a VPC
      network in the project.
    secrets: Names of secrets in Secret Manager.
    sslCa: Optional. SSL certificate to use for requests to GitHub Enterprise.
    webhookKey: The key that should be attached to webhook calls to the
      ReceiveWebhook endpoint.
  """
    appConfigJson = _messages.MessageField('GCSLocation', 1)
    appId = _messages.IntegerField(2)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    hostUrl = _messages.StringField(5)
    name = _messages.StringField(6)
    peeredNetwork = _messages.StringField(7)
    secrets = _messages.MessageField('GitHubEnterpriseSecrets', 8)
    sslCa = _messages.StringField(9)
    webhookKey = _messages.StringField(10)