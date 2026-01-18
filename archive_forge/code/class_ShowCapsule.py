from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class ShowCapsule(command.ShowOne):
    """Show a capsule"""
    log = logging.getLogger(__name__ + '.ShowCapsule')

    def get_parser(self, prog_name):
        parser = super(ShowCapsule, self).get_parser(prog_name)
        parser.add_argument('capsule', metavar='<capsule>', help='ID or name of the capsule to show.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.capsule
        opts = zun_utils.remove_null_parms(**opts)
        capsule = client.capsules.get(**opts)
        zun_utils.format_container_addresses(capsule)
        columns = _capsule_columns(capsule)
        return (columns, utils.get_item_properties(capsule, columns))