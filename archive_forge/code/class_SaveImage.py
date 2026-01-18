import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class SaveImage(command.Command):
    _description = _('Save an image locally')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--file', metavar='<filename>', dest='filename', help=_('Downloaded image save filename (default: stdout)'))
        parser.add_argument('image', metavar='<image>', help=_('Image to save (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        output_file = parsed_args.filename
        if output_file is None:
            output_file = getattr(sys.stdout, 'buffer', sys.stdout)
        image_client.download_image(image.id, stream=True, output=output_file)