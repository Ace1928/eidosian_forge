import copy
import datetime
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListCachedImage(command.Lister):
    _description = _('Get Cache State')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        data = _format_image_cache(dict(image_client.get_image_cache()))
        columns = ['image_id', 'state', 'last_accessed', 'last_modified', 'size', 'hits']
        column_headers = ['ID', 'State', 'Last Accessed (UTC)', 'Last Modified (UTC)', 'Size', 'Hits']
        return (column_headers, (utils.get_dict_properties(image, columns) for image in data))