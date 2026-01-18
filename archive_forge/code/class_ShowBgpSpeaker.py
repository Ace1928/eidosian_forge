from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class ShowBgpSpeaker(command.ShowOne):
    _description = _('Show a BGP speaker')

    def get_parser(self, prog_name):
        parser = super(ShowBgpSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='<bgp-speaker>', help=_('BGP speaker to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgp_speaker(parsed_args.bgp_speaker, ignore_missing=False).id
        obj = client.get_bgp_speaker(id)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)