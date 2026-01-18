import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
class TemplateDelete(command.Command):
    """Delete a template"""

    def get_parser(self, prog_name):
        parser = super(TemplateDelete, self).get_parser(prog_name)
        parser.add_argument('id', help='<ID or Name> of a template', nargs='+')
        parser.add_argument('--wait', type=int, default=None, nargs='?', const=sys.maxsize, help='Wait until template is DELETED or in ERROR default is to wait forever else number of seconds')
        return parser

    @property
    def formatter_default(self):
        return 'json'

    def take_action(self, parsed_args):
        ids = parsed_args.id
        wait = parsed_args.wait
        utils.get_client(self).template.delete(ids=ids)
        if wait:
            utils.wait_for_action_to_end(wait, self._check_deleted, ids=ids)

    def _check_deleted(self, ids):
        for _id in ids:
            try:
                utils.get_client(self).template.show(_id)
            except Exception:
                pass
            else:
                return False
        return True