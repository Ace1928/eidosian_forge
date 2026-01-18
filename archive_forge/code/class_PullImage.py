from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class PullImage(command.ShowOne):
    """Pull specified image into a host"""
    log = logging.getLogger(__name__ + '.PullImage')

    def get_parser(self, prog_name):
        parser = super(PullImage, self).get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help='Name of the image')
        parser.add_argument('host', metavar='<host>', help='Name or UUID of the host')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['repo'] = parsed_args.image
        opts['host'] = parsed_args.host
        image = client.images.create(**opts)
        columns = _image_columns(image)
        return (columns, utils.get_item_properties(image, columns))