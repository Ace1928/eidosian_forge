from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ValidateMediaAssetMessage(message):
    """Validate all parsed message from file are valid."""
    errors = encoding.UnrecognizedFieldIter(message)
    unrecognized_field_paths = []
    for edges_to_message, field_names in errors:
        message_field_path = '.'.join((six.text_type(e) for e in edges_to_message))
        for field_name in field_names:
            unrecognized_field_paths.append('{}.{}'.format(message_field_path, field_name))
    if unrecognized_field_paths:
        error_msg_lines = ['Invalid schema, the following fields are unrecognized:'] + unrecognized_field_paths
        raise exceptions.Error('\n'.join(error_msg_lines))