import copy
import datetime
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class QueueCachedImage(command.Command):
    _description = _('Queue image(s) for caching.')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('images', metavar='<image>', nargs='+', help=_('Image to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        failures = 0
        for image in parsed_args.images:
            try:
                image_obj = image_client.find_image(image, ignore_missing=False)
                image_client.queue_image(image_obj.id)
            except Exception as e:
                failures += 1
                msg = _("Failed to queue image with name or ID '%(image)s': %(e)s")
                LOG.error(msg, {'image': image, 'e': e})
        if failures > 0:
            total = len(parsed_args.images)
            msg = _('Failed to queue %(failures)s of %(total)s images') % {'failures': failures, 'total': total}
            raise exceptions.CommandError(msg)