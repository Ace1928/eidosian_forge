from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _GetSingleConfigToMaskMapping(self, config, prefix):
    """Build a map from each arg and its clear_ counterpart to a mask field."""
    fields_to_mask = dict()
    for field in config.keys():
        output_field = config[field]
        if prefix:
            fields_to_mask[field] = '{}.{}'.format(prefix, output_field)
        else:
            fields_to_mask[field] = output_field
        fields_to_mask[_EquivalentClearArg(field)] = fields_to_mask[field]
    return fields_to_mask