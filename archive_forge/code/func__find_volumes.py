import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _find_volumes(parsed_args_volumes, volume_client):
    result = 0
    uuid = ''
    for volume in parsed_args_volumes:
        try:
            volume_id = utils.find_resource(volume_client.volumes, volume).id
            uuid += volume_id + ','
        except Exception as e:
            result += 1
            LOG.error(_("Failed to find volume with name or ID '%(volume)s':%(e)s") % {'volume': volume, 'e': e})
    return (result, uuid)