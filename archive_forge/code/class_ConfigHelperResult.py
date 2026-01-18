from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core.credentials import creds as c_creds
class ConfigHelperResult(object):
    """The result of the gcloud config config-helper command that gets serialzied.

  Attributes:
    credential: Credential, The OAuth2 credential information.
    configuration: Configuration, Local Cloud SDK configuration information.
    sentinels: Sentinels, Paths to various sentinel files.
  """

    def __init__(self, credential, active_configuration, properties):
        self.credential = Credential(credential)
        self.configuration = Configuration(active_configuration, properties)
        self.sentinels = Sentinels(config.Paths().config_sentinel_file)