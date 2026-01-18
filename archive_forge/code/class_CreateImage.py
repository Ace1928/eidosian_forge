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
class CreateImage(command.ShowOne):
    _description = _('Create/upload an image')
    deadopts = ('size', 'location', 'copy-from', 'checksum', 'store')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('name', metavar='<image-name>', help=_('New image name'))
        parser.add_argument('--id', metavar='<id>', help=_('Image ID to reserve'))
        parser.add_argument('--container-format', default=DEFAULT_CONTAINER_FORMAT, choices=CONTAINER_CHOICES, metavar='<container-format>', help=_('Image container format. The supported options are: %(option_list)s. The default format is: %(default_opt)s') % {'option_list': ', '.join(CONTAINER_CHOICES), 'default_opt': DEFAULT_CONTAINER_FORMAT})
        parser.add_argument('--disk-format', default=DEFAULT_DISK_FORMAT, choices=DISK_CHOICES, metavar='<disk-format>', help=_('Image disk format. The supported options are: %s. The default format is: raw') % ', '.join(DISK_CHOICES))
        parser.add_argument('--min-disk', metavar='<disk-gb>', type=int, help=_('Minimum disk size needed to boot image, in gigabytes'))
        parser.add_argument('--min-ram', metavar='<ram-mb>', type=int, help=_('Minimum RAM size needed to boot image, in megabytes'))
        source_group = parser.add_mutually_exclusive_group()
        source_group.add_argument('--file', dest='filename', metavar='<file>', help=_('Upload image from local file'))
        source_group.add_argument('--volume', metavar='<volume>', help=_('Create image from a volume'))
        parser.add_argument('--force', dest='force', action='store_true', default=False, help=_('Force image creation if volume is in use (only meaningful with --volume)'))
        parser.add_argument('--progress', action='store_true', default=False, help=_('Show upload progress bar (ignored if passing data via stdin)'))
        parser.add_argument('--sign-key-path', metavar='<sign-key-path>', default=[], help=_('Sign the image using the specified private key. Only use in combination with --sign-cert-id'))
        parser.add_argument('--sign-cert-id', metavar='<sign-cert-id>', default=[], help=_('The specified certificate UUID is a reference to the certificate in the key manager that corresponds to the public key and is used for signature validation. Only use in combination with --sign-key-path'))
        _add_is_protected_args(parser)
        _add_visibility_args(parser)
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a property on this image (repeat option to set multiple properties)'))
        parser.add_argument('--tag', dest='tags', metavar='<tag>', action='append', help=_('Set a tag on this image (repeat option to set multiple tags)'))
        parser.add_argument('--project', metavar='<project>', help=_('Set an alternate project on this image (name or ID)'))
        parser.add_argument('--import', dest='use_import', action='store_true', help=_('Force the use of glance image import instead of direct upload'))
        identity_common.add_project_domain_option_to_parser(parser)
        for deadopt in self.deadopts:
            parser.add_argument('--%s' % deadopt, metavar='<%s>' % deadopt, dest=deadopt.replace('-', '_'), help=argparse.SUPPRESS)
        return parser

    def _take_action_image(self, parsed_args):
        identity_client = self.app.client_manager.identity
        image_client = self.app.client_manager.image
        kwargs = {'allow_duplicates': True}
        copy_attrs = ('name', 'id', 'container_format', 'disk_format', 'min_disk', 'min_ram', 'tags', 'visibility')
        for attr in copy_attrs:
            if attr in parsed_args:
                val = getattr(parsed_args, attr, None)
                if val:
                    kwargs[attr] = val
        if getattr(parsed_args, 'properties', None):
            for k, v in parsed_args.properties.items():
                kwargs[k] = str(v)
        if parsed_args.is_protected is not None:
            kwargs['is_protected'] = parsed_args.is_protected
        if parsed_args.visibility is not None:
            kwargs['visibility'] = parsed_args.visibility
        if parsed_args.project:
            kwargs['owner_id'] = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        if parsed_args.use_import:
            kwargs['use_import'] = True
        if parsed_args.filename:
            try:
                fp = open(parsed_args.filename, 'rb')
            except FileNotFoundError:
                raise exceptions.CommandError('%r is not a valid file' % parsed_args.filename)
        else:
            fp = get_data_from_stdin()
        if fp is not None and parsed_args.volume:
            msg = _('Uploading data and using container are not allowed at the same time')
            raise exceptions.CommandError(msg)
        if parsed_args.progress and parsed_args.filename:
            filesize = os.path.getsize(parsed_args.filename)
            if filesize is not None:
                kwargs['validate_checksum'] = False
                kwargs['data'] = progressbar.VerboseFileWrapper(fp, filesize)
            else:
                kwargs['data'] = fp
        elif parsed_args.filename:
            kwargs['filename'] = parsed_args.filename
        elif fp:
            kwargs['validate_checksum'] = False
            kwargs['data'] = fp
        if parsed_args.sign_key_path or parsed_args.sign_cert_id:
            if not parsed_args.filename:
                msg = _('signing an image requires the --file option, passing files via stdin when signing is not supported.')
                raise exceptions.CommandError(msg)
            if len(parsed_args.sign_key_path) < 1 or len(parsed_args.sign_cert_id) < 1:
                msg = _("'sign-key-path' and 'sign-cert-id' must both be specified when attempting to sign an image.")
                raise exceptions.CommandError(msg)
            sign_key_path = parsed_args.sign_key_path
            sign_cert_id = parsed_args.sign_cert_id
            signer = image_signer.ImageSigner()
            try:
                pw = utils.get_password(self.app.stdin, prompt='Please enter private key password, leave empty if none: ', confirm=False)
                if not pw or len(pw) < 1:
                    pw = None
                else:
                    pw = pw.encode()
                signer.load_private_key(sign_key_path, password=pw)
            except Exception:
                msg = _('Error during sign operation: private key could not be loaded.')
                raise exceptions.CommandError(msg)
            signature = signer.generate_signature(fp)
            signature_b64 = b64encode(signature)
            kwargs['img_signature'] = signature_b64
            kwargs['img_signature_certificate_uuid'] = sign_cert_id
            kwargs['img_signature_hash_method'] = signer.hash_method
            if signer.padding_method:
                kwargs['img_signature_key_type'] = signer.padding_method
        image = image_client.create_image(**kwargs)
        if parsed_args.filename:
            fp.close()
        return _format_image(image)

    def _take_action_volume(self, parsed_args):
        volume_client = self.app.client_manager.volume
        unsupported_opts = {'id', 'min_disk', 'min_ram', 'file', 'force', 'progress', 'sign_key_path', 'sign_cert_id', 'properties', 'tags', 'project', 'use_import'}
        for unsupported_opt in unsupported_opts:
            if getattr(parsed_args, unsupported_opt, None):
                opt_name = unsupported_opt.replace('-', '_')
                if unsupported_opt == 'use_import':
                    opt_name = 'import'
                msg = _("'--%s' was given, which is not supported when creating an image from a volume. This will be an error in a future version.")
                LOG.warning(msg % opt_name)
        source_volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        kwargs = {}
        if volume_client.api_version < api_versions.APIVersion('3.1'):
            if parsed_args.visibility or parsed_args.is_protected is not None:
                msg = _('--os-volume-api-version 3.1 or greater is required to support the --public, --private, --community, --shared or --protected option.')
                raise exceptions.CommandError(msg)
        else:
            kwargs.update(visibility=parsed_args.visibility or 'private', protected=parsed_args.is_protected or False)
        response, body = volume_client.volumes.upload_to_image(source_volume.id, parsed_args.force, parsed_args.name, parsed_args.container_format, parsed_args.disk_format, **kwargs)
        info = body['os-volume_upload_image']
        try:
            info['volume_type'] = info['volume_type']['name']
        except TypeError:
            info['volume_type'] = None
        return info

    def take_action(self, parsed_args):
        for deadopt in self.deadopts:
            if getattr(parsed_args, deadopt.replace('-', '_'), None):
                msg = _('ERROR: --%s was given, which is an Image v1 option that is no longer supported in Image v2')
                raise exceptions.CommandError(msg % deadopt)
        if parsed_args.volume:
            info = self._take_action_volume(parsed_args)
        else:
            info = self._take_action_image(parsed_args)
        return zip(*sorted(info.items()))