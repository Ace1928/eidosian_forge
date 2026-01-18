from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def BuildGoogleChannelConfig(self, google_channel_config_name, crypto_key_name):
    return self._messages.GoogleChannelConfig(name=google_channel_config_name, cryptoKeyName=crypto_key_name)