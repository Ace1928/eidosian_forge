import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
class TemplateVersions(lister.Lister):
    """List all template versions"""

    def get_parser(self, prog_name):
        parser = super(TemplateVersions, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        templates = utils.get_client(self).template.versions()
        return utils.list2cols_with_rename((('Version', 'version'), ('Status', 'status')), templates)