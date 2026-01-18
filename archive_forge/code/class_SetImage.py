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
class SetImage(command.Command):
    _description = _('Set image properties')
    deadopts = ('visibility',)

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help=_('Image to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('New image name'))
        parser.add_argument('--min-disk', type=int, metavar='<disk-gb>', help=_('Minimum disk size needed to boot image, in gigabytes'))
        parser.add_argument('--min-ram', type=int, metavar='<ram-mb>', help=_('Minimum RAM size needed to boot image, in megabytes'))
        parser.add_argument('--container-format', metavar='<container-format>', choices=CONTAINER_CHOICES, help=_('Image container format. The supported options are: %s') % ', '.join(CONTAINER_CHOICES))
        parser.add_argument('--disk-format', metavar='<disk-format>', choices=DISK_CHOICES, help=_('Image disk format. The supported options are: %s') % ', '.join(DISK_CHOICES))
        _add_is_protected_args(parser)
        _add_visibility_args(parser)
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a property on this image (repeat option to set multiple properties)'))
        parser.add_argument('--tag', dest='tags', metavar='<tag>', default=None, action='append', help=_('Set a tag on this image (repeat option to set multiple tags)'))
        parser.add_argument('--architecture', metavar='<architecture>', help=_('Operating system architecture'))
        parser.add_argument('--instance-id', metavar='<instance-id>', help=_('ID of server instance used to create this image'))
        parser.add_argument('--instance-uuid', metavar='<instance-id>', dest='instance_id', help=argparse.SUPPRESS)
        parser.add_argument('--kernel-id', metavar='<kernel-id>', help=_('ID of kernel image used to boot this disk image'))
        parser.add_argument('--os-distro', metavar='<os-distro>', help=_('Operating system distribution name'))
        parser.add_argument('--os-version', metavar='<os-version>', help=_('Operating system distribution version'))
        parser.add_argument('--ramdisk-id', metavar='<ramdisk-id>', help=_('ID of ramdisk image used to boot this disk image'))
        deactivate_group = parser.add_mutually_exclusive_group()
        deactivate_group.add_argument('--deactivate', action='store_true', help=_('Deactivate the image'))
        deactivate_group.add_argument('--activate', action='store_true', help=_('Activate the image'))
        parser.add_argument('--project', metavar='<project>', help=_('Set an alternate project on this image (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        for deadopt in self.deadopts:
            parser.add_argument('--%s' % deadopt, metavar='<%s>' % deadopt, dest=f'dead_{deadopt.replace('-', '_')}', help=argparse.SUPPRESS)
        membership_group = parser.add_mutually_exclusive_group()
        membership_group.add_argument('--accept', action='store_const', const='accepted', dest='membership', default=None, help=_("Accept the image membership for either the project indicated by '--project', if provided, or the current user's project"))
        membership_group.add_argument('--reject', action='store_const', const='rejected', dest='membership', default=None, help=_("Reject the image membership for either the project indicated by '--project', if provided, or the current user's project"))
        membership_group.add_argument('--pending', action='store_const', const='pending', dest='membership', default=None, help=_("Reset the image membership to 'pending'"))
        hidden_group = parser.add_mutually_exclusive_group()
        hidden_group.add_argument('--hidden', dest='is_hidden', default=None, action='store_true', help=_('Hide the image'))
        hidden_group.add_argument('--unhidden', dest='is_hidden', default=None, action='store_false', help=_('Unhide the image'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        image_client = self.app.client_manager.image
        for deadopt in self.deadopts:
            if getattr(parsed_args, f'dead_{deadopt.replace('-', '_')}', None):
                raise exceptions.CommandError(_('ERROR: --%s was given, which is an Image v1 option that is no longer supported in Image v2') % deadopt)
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        activation_status = None
        if parsed_args.deactivate or parsed_args.activate:
            if parsed_args.deactivate:
                image_client.deactivate_image(image.id)
                activation_status = 'deactivated'
            if parsed_args.activate:
                image_client.reactivate_image(image.id)
                activation_status = 'activated'
        if parsed_args.membership:
            if not project_id:
                project_id = self.app.client_manager.auth_ref.project_id
            image_client.update_member(image=image.id, member=project_id, status=parsed_args.membership)
        kwargs = {}
        copy_attrs = ('architecture', 'container_format', 'disk_format', 'file', 'instance_id', 'kernel_id', 'locations', 'min_disk', 'min_ram', 'name', 'os_distro', 'os_version', 'prefix', 'progress', 'ramdisk_id', 'tags', 'visibility')
        for attr in copy_attrs:
            if attr in parsed_args:
                val = getattr(parsed_args, attr, None)
                if val is not None:
                    kwargs[attr] = val
        if getattr(parsed_args, 'properties', None):
            for k, v in parsed_args.properties.items():
                kwargs[k] = str(v)
        if parsed_args.is_protected is not None:
            kwargs['is_protected'] = parsed_args.is_protected
        if parsed_args.visibility is not None:
            kwargs['visibility'] = parsed_args.visibility
        if parsed_args.project:
            kwargs['owner_id'] = project_id
        if parsed_args.tags:
            kwargs['tags'] = list(set(image.tags).union(set(parsed_args.tags)))
        if parsed_args.is_hidden is not None:
            kwargs['is_hidden'] = parsed_args.is_hidden
        try:
            image = image_client.update_image(image.id, **kwargs)
        except Exception:
            if activation_status is not None:
                LOG.info(_('Image %(id)s was %(status)s.'), {'id': image.id, 'status': activation_status})
            raise