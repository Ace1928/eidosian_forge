from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowQuotaDefaults(command.ShowOne):
    """Show quota defaults"""

    def take_action(self, parsed_args):
        rows = const.QUOTA_ROWS
        data = self.app.client_manager.load_balancer.quota_defaults_show()
        return (rows, utils.get_dict_properties(data['quota'], rows))