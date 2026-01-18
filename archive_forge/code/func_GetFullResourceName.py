from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetFullResourceName(resource_ref):
    """Convert a full resource URL to a full resource name (FRN).

  See https://cloud.google.com/iam/docs/full-resource-names.

  Args:
    resource_ref: googlecloudsdk.core.resources.Resource.

  Returns:
    str: Full resource name of the resource
  """
    full_name = resource_ref.SelfLink()
    full_name = re.sub('\\w+://', '//', full_name)
    full_name = re.sub('/v[0-9]+[0-9a-zA-Z]*/', '/', full_name)
    universe_domain_property = properties.VALUES.core.universe_domain
    universe_domain = universe_domain_property.Get()
    if universe_domain_property.default != universe_domain:
        full_name = full_name.replace(universe_domain, universe_domain_property.default, 1)
    if full_name.startswith('//www.'):
        splitted_list = full_name.split('/')
        service = full_name.split('/')[3]
        splitted_list.pop(3)
        full_name = '/'.join(splitted_list)
        full_name = full_name.replace('//www.', '//{0}.'.format(service))
    return full_name