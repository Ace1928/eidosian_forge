import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class UnsetShareGroup(command.Command):
    """Unset a share group property."""
    _description = _('Unset a share group property')

    def get_parser(self, prog_name):
        parser = super(UnsetShareGroup, self).get_parser(prog_name)
        parser.add_argument('share_group', metavar='<share-group>', help=_('Name or ID of the share group to set a property for.'))
        parser.add_argument('--name', action='store_true', help=_('Unset share group name.'))
        parser.add_argument('--description', action='store_true', help=_('Unset share group description.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group = osc_utils.find_resource(share_client.share_groups, parsed_args.share_group)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = None
        if parsed_args.description:
            kwargs['description'] = None
        if kwargs:
            try:
                share_client.share_groups.update(share_group, **kwargs)
            except Exception as e:
                raise exceptions.CommandError(_('Failed to unset share_group name or description : %s' % e))