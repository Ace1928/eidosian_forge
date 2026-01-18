from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class DeleteCapsule(command.Command):
    """Delete specified capsule(s)"""
    log = logging.getLogger(__name__ + '.DeleteCapsule')

    def get_parser(self, prog_name):
        parser = super(DeleteCapsule, self).get_parser(prog_name)
        parser.add_argument('capsule', metavar='<capsule>', nargs='+', help='ID or name of the capsule(s) to delete.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        capsules = parsed_args.capsule
        for capsule in capsules:
            opts = {}
            opts['id'] = capsule
            try:
                client.capsules.delete(**opts)
                print(_('Request to delete capsule %s has been accepted.') % capsule)
            except Exception as e:
                print('Delete for capsule %(capsule)s failed: %(e)s' % {'capsule': capsule, 'e': e})