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
class StageImage(command.Command):
    _description = _('Upload data for a specific image to staging.\nThis requires support for the interoperable image import process, which was first introduced in Image API version 2.6 (Glance 16.0.0 (Queens))')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--file', metavar='<file>', dest='filename', help=_('Local file that contains disk image to be uploaded. Alternatively, images can be passed via stdin.'))
        parser.add_argument('--progress', action='store_true', default=False, help=_('Show upload progress bar (ignored if passing data via stdin)'))
        parser.add_argument('image', metavar='<image>', help=_('Image to upload data for (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        if parsed_args.filename:
            try:
                fp = open(parsed_args.filename, 'rb')
            except FileNotFoundError:
                raise exceptions.CommandError('%r is not a valid file' % parsed_args.filename)
        else:
            fp = get_data_from_stdin()
        kwargs = {}
        if parsed_args.progress and parsed_args.filename:
            filesize = os.path.getsize(parsed_args.filename)
            if filesize is not None:
                kwargs['data'] = progressbar.VerboseFileWrapper(fp, filesize)
            else:
                kwargs['data'] = fp
        elif parsed_args.filename:
            kwargs['filename'] = parsed_args.filename
        elif fp:
            kwargs['data'] = fp
        image_client.stage_image(image, **kwargs)