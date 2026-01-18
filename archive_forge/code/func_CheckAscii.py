from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def CheckAscii(s):
    """Check if a string is ascii."""
    try:
        s.decode('ascii')
    except UnicodeError as error:
        raise ValueError('Non-ascii characters [{0}] found in the current authorized view definition, please use --pre-encoded instead. [{1}].'.format(s, error))