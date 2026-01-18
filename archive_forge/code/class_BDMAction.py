import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class BDMAction(parseractions.MultiKeyValueAction):

    def __init__(self, option_strings, dest, **kwargs):
        required_keys = []
        optional_keys = ['uuid', 'source_type', 'destination_type', 'disk_bus', 'device_type', 'device_name', 'volume_size', 'guest_format', 'boot_index', 'delete_on_termination', 'tag', 'volume_type']
        super().__init__(option_strings, dest, required_keys=required_keys, optional_keys=optional_keys, **kwargs)

    def validate_keys(self, keys):
        """Validate the provided keys.

        :param keys: A list of keys to validate.
        """
        valid_keys = self.required_keys | self.optional_keys
        invalid_keys = [k for k in keys if k not in valid_keys]
        if invalid_keys:
            msg = _('Invalid keys %(invalid_keys)s specified.\nValid keys are: %(valid_keys)s')
            raise argparse.ArgumentError(self, msg % {'invalid_keys': ', '.join(invalid_keys), 'valid_keys': ', '.join(valid_keys)})
        missing_keys = [k for k in self.required_keys if k not in keys]
        if missing_keys:
            msg = _('Missing required keys %(missing_keys)s.\nRequired keys are: %(required_keys)s')
            raise argparse.ArgumentError(self, msg % {'missing_keys': ', '.join(missing_keys), 'required_keys': ', '.join(self.required_keys)})

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        if os.path.exists(values):
            with open(values) as fh:
                data = json.load(fh)
            self.validate_keys(list(data))
            getattr(namespace, self.dest, []).append(data)
        else:
            super().__call__(parser, namespace, values, option_string)