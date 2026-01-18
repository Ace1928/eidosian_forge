import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ShowShareAccess(command.ShowOne):
    """Display a share access rule."""
    _description = _('Display a share access rule. Available for API microversion 2.45 and higher')

    def get_parser(self, prog_name):
        parser = super(ShowShareAccess, self).get_parser(prog_name)
        parser.add_argument('access_id', metavar='<access_id>', help=_('ID of the NAS share access rule.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if share_client.api_version >= api_versions.APIVersion('2.45'):
            access_rule = share_client.share_access_rules.get(parsed_args.access_id)
            access_rule._info.update({'properties': utils.format_properties(access_rule._info.pop('metadata', {}))})
            return (ACCESS_RULE_ATTRIBUTES, oscutils.get_dict_properties(access_rule._info, ACCESS_RULE_ATTRIBUTES))
        else:
            raise exceptions.CommandError('Displaying share access rule details is only available with API microversion 2.45 and higher.')