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