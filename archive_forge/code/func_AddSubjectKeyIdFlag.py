from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def AddSubjectKeyIdFlag(parser):
    base.Argument('--subject-key-id', help='Optional field to specify subject key ID for certificate. DO NOT USE except to maintain a previously established identifier for a public key, whose SKI was not generated using method (1) described in RFC 5280 section 4.2.1.2.').AddToParser(parser)