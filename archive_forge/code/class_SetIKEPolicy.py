from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import utils as vpn_utils
class SetIKEPolicy(command.Command):
    _description = _('Set IKE policy properties')

    def get_parser(self, prog_name):
        parser = super(SetIKEPolicy, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument('--name', metavar='<name>', help=_('Name of the IKE policy'))
        parser.add_argument('ikepolicy', metavar='<ike-policy>', help=_('IKE policy to set (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        if parsed_args.name:
            attrs['name'] = parsed_args.name
        ike_id = client.find_vpn_ike_policy(parsed_args.ikepolicy, ignore_missing=False)['id']
        try:
            client.update_vpn_ike_policy(ike_id, **attrs)
        except Exception as e:
            msg = _("Failed to set IKE policy '%(ike)s': %(e)s") % {'ike': parsed_args.ikepolicy, 'e': e}
            raise exceptions.CommandError(msg)