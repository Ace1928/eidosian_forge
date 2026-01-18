import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _parse_block_device_mapping_v2(cs, args, image):
    bdm = []
    if args.boot_volume:
        bdm_dict = {'uuid': args.boot_volume, 'source_type': 'volume', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}
        bdm.append(bdm_dict)
    if args.snapshot:
        bdm_dict = {'uuid': args.snapshot, 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}
        bdm.append(bdm_dict)
    supports_volume_type = cs.api_version >= api_versions.APIVersion('2.67')
    for device_spec in args.block_device:
        spec_dict = _parse_device_spec(device_spec)
        bdm_dict = {}
        if 'tag' in spec_dict and (not _supports_block_device_tags(cs)):
            raise exceptions.CommandError(_("'tag' in block device mapping is not supported in API version %(version)s.") % {'version': cs.api_version.get_string()})
        if 'volume_type' in spec_dict and (not supports_volume_type):
            raise exceptions.CommandError(_("'volume_type' in block device mapping is not supported in API version %(version)s.") % {'version': cs.api_version.get_string()})
        for key, value in spec_dict.items():
            bdm_dict[CLIENT_BDM2_KEYS[key]] = value
        source_type = bdm_dict.get('source_type')
        if not source_type:
            bdm_dict['source_type'] = 'blank'
        elif source_type not in ('volume', 'image', 'snapshot', 'blank'):
            raise exceptions.CommandError(_("The value of source_type key of --block-device should be one of 'volume', 'image', 'snapshot' or 'blank' but it was '%(action)s'") % {'action': source_type})
        destination_type = bdm_dict.get('destination_type')
        if not destination_type:
            source_type = bdm_dict['source_type']
            if source_type in ('image', 'blank'):
                bdm_dict['destination_type'] = 'local'
            if source_type in ('snapshot', 'volume'):
                bdm_dict['destination_type'] = 'volume'
        elif destination_type not in ('local', 'volume'):
            raise exceptions.CommandError(_("The value of destination_type key of --block-device should be either 'local' or 'volume' but it was '%(action)s'") % {'action': destination_type})
        if 'delete_on_termination' in bdm_dict:
            action = bdm_dict['delete_on_termination']
            if action not in ['remove', 'preserve']:
                raise exceptions.CommandError(_("The value of shutdown key of --block-device shall be either 'remove' or 'preserve' but it was '%(action)s'") % {'action': action})
            bdm_dict['delete_on_termination'] = action == 'remove'
        elif bdm_dict.get('destination_type') == 'local':
            bdm_dict['delete_on_termination'] = True
        bdm.append(bdm_dict)
    for ephemeral_spec in args.ephemeral:
        bdm_dict = {'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True}
        try:
            eph_dict = _parse_device_spec(ephemeral_spec)
        except ValueError:
            err_msg = _("Invalid ephemeral argument '%s'.") % args.ephemeral
            raise argparse.ArgumentTypeError(err_msg)
        if 'size' in eph_dict:
            bdm_dict['volume_size'] = eph_dict['size']
        if 'format' in eph_dict:
            bdm_dict['guest_format'] = eph_dict['format']
        bdm.append(bdm_dict)
    if args.swap:
        bdm_dict = {'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True, 'guest_format': 'swap', 'volume_size': args.swap}
        bdm.append(bdm_dict)
    return bdm