import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class UnsetShareGroupType(command.Command):
    """Unset share group type extra specs."""
    _description = _('Unset share group type extra specs')
    log = logging.getLogger(__name__ + '.UnsetShareGroupType')

    def get_parser(self, prog_name):
        parser = super(UnsetShareGroupType, self).get_parser(prog_name)
        parser.add_argument('share_group_type', metavar='<share-group-type>', help=_('Name or ID of the share grouptype to modify'))
        parser.add_argument('group_specs', metavar='<key>', nargs='+', help=_('Remove group specs from this share group type'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        try:
            share_group_type_obj = apiutils.find_resource(share_client.share_group_types, parsed_args.share_group_type)
        except Exception as e:
            msg = LOG.error(_("Failed to find the share group type with name or ID '%(share_group_type)s': %(e)s"), {'share_group_type': parsed_args.share_group_type, 'e': e})
            raise exceptions.CommandError(msg)
        if parsed_args.group_specs:
            try:
                share_group_type_obj.unset_keys(parsed_args.group_specs)
            except Exception as e:
                raise exceptions.CommandError('Failed to remove share type group extra spec: %s' % e)