import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class SetShareGroupType(command.Command):
    """Set share type properties."""
    _description = _('Set share group type properties')
    log = logging.getLogger(__name__ + '.SetShareGroupType')

    def get_parser(self, prog_name):
        parser = super(SetShareGroupType, self).get_parser(prog_name)
        parser.add_argument('share_group_type', metavar='<share-group-type>', help=_('Name or ID of the share group type to modify'))
        parser.add_argument('--group-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Extra specs key and value of share group type that will be used for share type creation. OPTIONAL: Default=None. Example: --group-specs consistent-snapshot-support=True'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        try:
            share_group_type_obj = apiutils.find_resource(share_client.share_group_types, parsed_args.share_group_type)
        except Exception as e:
            msg = LOG.error(_("Failed to find the share group type with name or ID '%(share_group_type)s': %(e)s"), {'share_group_type': parsed_args.share_group_type, 'e': e})
            raise exceptions.CommandError(msg)
        kwargs = {}
        if kwargs:
            share_group_type_obj.set_keys(**kwargs)
        if parsed_args.group_specs:
            group_specs = utils.extract_group_specs(extra_specs={}, specs_to_add=parsed_args.group_specs)
            try:
                share_group_type_obj.set_keys(group_specs)
            except Exception as e:
                raise exceptions.CommandError('Failed to set share group type key: %s' % e)