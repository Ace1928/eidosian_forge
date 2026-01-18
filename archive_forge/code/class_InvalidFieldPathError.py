from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
class InvalidFieldPathError(Error):
    """The referenced field path could not be found in the message object."""

    def __init__(self, field_path, message, reason):
        super(InvalidFieldPathError, self).__init__('Invalid field path [{}] for message [{}]. Details: [{}]'.format(field_path, _GetFullClassName(message), reason))