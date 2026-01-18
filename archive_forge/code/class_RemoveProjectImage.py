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
class RemoveProjectImage(command.Command):
    _description = _('Disassociate project with image')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help=_('Image to unshare (name or ID)'))
        parser.add_argument('project', metavar='<project>', help=_('Project to disassociate with image (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        identity_client = self.app.client_manager.identity
        project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        image_client.remove_member(member=project_id, image=image.id)