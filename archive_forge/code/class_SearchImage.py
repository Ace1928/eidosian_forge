from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class SearchImage(command.Lister):
    """Search specified image"""
    log = logging.getLogger(__name__ + '.SearchImage')

    def get_parser(self, prog_name):
        parser = super(SearchImage, self).get_parser(prog_name)
        parser.add_argument('--image-driver', metavar='<image-driver>', help='Name of the image driver')
        parser.add_argument('image_name', metavar='<image_name>', help='Name of the image')
        parser.add_argument('--exact-match', default=False, action='store_true', help='exact match image name')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['image_driver'] = parsed_args.image_driver
        opts['image'] = parsed_args.image_name
        opts['exact_match'] = parsed_args.exact_match
        opts = zun_utils.remove_null_parms(**opts)
        images = client.images.search_image(**opts)
        columns = ('ID', 'Name', 'Tags', 'Status', 'Size', 'Metadata')
        return (columns, (utils.get_item_properties(image, columns) for image in images))