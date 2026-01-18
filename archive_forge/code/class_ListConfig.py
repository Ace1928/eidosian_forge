import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import template_format
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class ListConfig(command.Lister):
    """List software configs"""
    log = logging.getLogger(__name__ + '.ListConfig')

    def get_parser(self, prog_name):
        parser = super(ListConfig, self).get_parser(prog_name)
        parser.add_argument('--limit', metavar='<limit>', help=_('Limit the number of configs returned'))
        parser.add_argument('--marker', metavar='<id>', help=_('Return configs that appear after the given config ID'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _list_config(heat_client, parsed_args)