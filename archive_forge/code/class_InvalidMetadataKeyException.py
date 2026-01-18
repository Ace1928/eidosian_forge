from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
class InvalidMetadataKeyException(Error):
    """InvalidMetadataKeyException is for not allowed metadata keys."""

    def __init__(self, metadata_key):
        super(InvalidMetadataKeyException, self).__init__('Metadata key "{0}" is not allowed when running containerized VM.'.format(metadata_key))