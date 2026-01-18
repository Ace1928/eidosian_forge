import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class DeleteShareGroupType(command.Command):
    """Delete a share group type."""
    _description = _('Delete a share group type')
    log = logging.getLogger(__name__ + '.DeleteShareGroupType')

    def get_parser(self, prog_name):
        parser = super(DeleteShareGroupType, self).get_parser(prog_name)
        parser.add_argument('share_group_types', metavar='<share-group-types>', nargs='+', help=_('Name or ID of the share group type(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share_group_type in parsed_args.share_group_types:
            try:
                share_group_type_obj = apiutils.find_resource(share_client.share_group_types, share_group_type)
                share_client.share_group_types.delete(share_group_type_obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete share group type with name or ID '%(share_group_type)s': %(e)s"), {'share_group_type': share_group_type, 'e': e})
        if result > 0:
            total = len(parsed_args.share_group_types)
            msg = _('%(result)s of %(total)s share group types failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)