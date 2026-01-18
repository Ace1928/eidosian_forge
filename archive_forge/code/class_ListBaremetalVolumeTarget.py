import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ListBaremetalVolumeTarget(command.Lister):
    """List baremetal volume targets."""
    log = logging.getLogger(__name__ + '.ListBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(ListBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('--node', dest='node', metavar='<node>', help=_('Only list volume targets of this node (name or UUID).'))
        parser.add_argument('--limit', dest='limit', metavar='<limit>', type=int, help=_('Maximum number of volume targets to return per request, 0 for no limit. Default is the maximum number used by the Baremetal API Service.'))
        parser.add_argument('--marker', dest='marker', metavar='<volume target>', help=_('Volume target UUID (for example, of the last volume target in the list from a previous request). Returns the list of volume targets after this UUID.'))
        parser.add_argument('--sort', dest='sort', metavar='<key>[:<direction>]', help=_('Sort output by specified volume target fields and directions (asc or desc) (default:asc). Multiple fields and directions can be specified, separated by comma.'))
        display_group = parser.add_mutually_exclusive_group(required=False)
        display_group.add_argument('--long', dest='detail', action='store_true', default=False, help=_('Show detailed information about volume targets.'))
        display_group.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.VOLUME_TARGET_DETAILED_RESOURCE.fields, help=_("One or more volume target fields. Only these fields will be fetched from the server. Can not be used when '--long' is specified."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)' % parsed_args)
        client = self.app.client_manager.baremetal
        columns = res_fields.VOLUME_TARGET_RESOURCE.fields
        labels = res_fields.VOLUME_TARGET_RESOURCE.labels
        params = {}
        if parsed_args.limit is not None and parsed_args.limit < 0:
            raise exc.CommandError(_('Expected non-negative --limit, got %s') % parsed_args.limit)
        params['limit'] = parsed_args.limit
        params['marker'] = parsed_args.marker
        if parsed_args.node is not None:
            params['node'] = parsed_args.node
        if parsed_args.detail:
            params['detail'] = parsed_args.detail
            columns = res_fields.VOLUME_TARGET_DETAILED_RESOURCE.fields
            labels = res_fields.VOLUME_TARGET_DETAILED_RESOURCE.labels
        elif parsed_args.fields:
            params['detail'] = False
            fields = itertools.chain.from_iterable(parsed_args.fields)
            resource = res_fields.Resource(list(fields))
            columns = resource.fields
            labels = resource.labels
            params['fields'] = columns
        self.log.debug('params(%s)' % params)
        data = client.volume_target.list(**params)
        data = oscutils.sort_items(data, parsed_args.sort)
        return (labels, (oscutils.get_item_properties(s, columns, formatters={'Properties': oscutils.format_dict}) for s in data))