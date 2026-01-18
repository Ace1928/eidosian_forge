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