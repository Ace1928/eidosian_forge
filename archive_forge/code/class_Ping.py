import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class Ping(command.ShowOne):
    """Check if Zaqar server is alive or not"""
    _description = _('Check if Zaqar server is alive or not')
    log = logging.getLogger(__name__ + '.Ping')

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        columns = ('Pingable',)
        return (columns, utils.get_dict_properties({'pingable': client.ping()}, columns))