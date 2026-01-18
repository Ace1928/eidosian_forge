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
def AddIgnoreDependentResourcesFlag(parser):
    base.Argument('--ignore-dependent-resources', help='This field skips the integrity check that would normally prevent breaking a CA Pool if it is used by another cloud resource and allows the CA Pool to be in a state where it is not able to issue certificates. Doing so may result in unintended and unrecoverable effects on any dependent resource(s) since the CA Pool would not be able to issue certificates.', action='store_true', default=False, required=False).AddToParser(parser)