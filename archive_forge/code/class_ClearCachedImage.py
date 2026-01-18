import copy
import datetime
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ClearCachedImage(command.Command):
    _description = _('Clear all images from cache, queue or both')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--cache', action='store_const', const='cache', dest='target', help=_('Clears all the cached images'))
        parser.add_argument('--queue', action='store_const', const='queue', dest='target', help=_('Clears all the queued images'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        target = parsed_args.target
        try:
            image_client.clear_cache(target)
        except Exception:
            msg = _('Failed to clear image cache')
            LOG.error(msg)
            raise exceptions.CommandError(msg)