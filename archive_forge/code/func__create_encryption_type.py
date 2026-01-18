import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _create_encryption_type(volume_client, volume_type, parsed_args):
    if not parsed_args.encryption_provider:
        msg = _("'--encryption-provider' should be specified while creating a new encryption type")
        raise exceptions.CommandError(msg)
    control_location = 'front-end'
    if parsed_args.encryption_control_location:
        control_location = parsed_args.encryption_control_location
    body = {'provider': parsed_args.encryption_provider, 'cipher': parsed_args.encryption_cipher, 'key_size': parsed_args.encryption_key_size, 'control_location': control_location}
    encryption = volume_client.volume_encryption_types.create(volume_type, body)
    return encryption