import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class UnsetShareAccess(command.Command):
    """Unset properties of share access rule."""
    _description = _('Unset properties of share access rule. Available for API microversion 2.45 and higher')

    def get_parser(self, prog_name):
        parser = super(UnsetShareAccess, self).get_parser(prog_name)
        parser.add_argument('access_id', metavar='<access_id>', help=_('ID of the NAS share access rule.'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Remove property from share access rule. (Repeat option to remove multiple properties) Available only for API microversion >= 2.45.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if share_client.api_version >= api_versions.APIVersion('2.45'):
            access_rule = share_client.share_access_rules.get(parsed_args.access_id)
            if parsed_args.property:
                try:
                    share_client.share_access_rules.unset_metadata(access_rule, parsed_args.property)
                except Exception as e:
                    raise exceptions.CommandError("Failed to unset properties for share access rule with ID '%s': %s" % (access_rule.id, e))
            else:
                raise exceptions.CommandError("Please specify '--property <key>' to unset a property. ")
        else:
            raise exceptions.CommandError('Option to unset properties of access rule is available only for API microversion 2.45 and higher')