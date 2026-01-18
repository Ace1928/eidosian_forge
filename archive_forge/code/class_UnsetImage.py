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
class UnsetImage(command.Command):
    _description = _('Unset image tags and properties')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help=_('Image to modify (name or ID)'))
        parser.add_argument('--tag', dest='tags', metavar='<tag>', default=[], action='append', help=_('Unset a tag on this image (repeat option to unset multiple tags)'))
        parser.add_argument('--property', dest='properties', metavar='<property-key>', default=[], action='append', help=_('Unset a property on this image (repeat option to unset multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        kwargs = {}
        tagret = 0
        propret = 0
        if parsed_args.tags:
            for k in parsed_args.tags:
                try:
                    image_client.remove_tag(image.id, k)
                except Exception:
                    LOG.error(_("tag unset failed, '%s' is a nonexistent tag "), k)
                    tagret += 1
        if parsed_args.properties:
            for k in parsed_args.properties:
                if k in image:
                    delattr(image, k)
                elif k in image.properties:
                    new_props = kwargs.get('properties', image.get('properties').copy())
                    new_props.pop(k, None)
                    kwargs['properties'] = new_props
                else:
                    LOG.error(_("property unset failed, '%s' is a nonexistent property "), k)
                    propret += 1
            image_client.update_image(image, **kwargs)
        tagtotal = len(parsed_args.tags)
        proptotal = len(parsed_args.properties)
        if tagret > 0 and propret > 0:
            msg = _('Failed to unset %(tagret)s of %(tagtotal)s tags,Failed to unset %(propret)s of %(proptotal)s properties.') % {'tagret': tagret, 'tagtotal': tagtotal, 'propret': propret, 'proptotal': proptotal}
            raise exceptions.CommandError(msg)
        elif tagret > 0:
            msg = _('Failed to unset %(tagret)s of %(tagtotal)s tags.') % {'tagret': tagret, 'tagtotal': tagtotal}
            raise exceptions.CommandError(msg)
        elif propret > 0:
            msg = _('Failed to unset %(propret)s of %(proptotal)s properties.') % {'propret': propret, 'proptotal': proptotal}
            raise exceptions.CommandError(msg)