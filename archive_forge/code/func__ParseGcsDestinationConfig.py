from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _ParseGcsDestinationConfig(self, gcs_destination_config_file, release_track):
    """Parses a GcsDestinationConfig into the GcsDestinationConfig message."""
    if release_track == base.ReleaseTrack.BETA:
        return self._ParseGcsDestinationConfigBeta(gcs_destination_config_file)
    return util.ParseMessageAndValidateSchema(gcs_destination_config_file, 'GcsDestinationConfig', self._messages.GcsDestinationConfig)