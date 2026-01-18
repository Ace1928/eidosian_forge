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
def _get_member_columns(item):
    column_map = {'image_id': 'image_id'}
    hidden_columns = ['id', 'location', 'name']
    return utils.get_osc_show_columns_for_sdk_resource(item.to_dict(), column_map, hidden_columns)