from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def ParseSecureMultiTenancyUserServiceAccountMappingString(user_service_account_mapping_string):
    """Parses a secure-multi-tenancy-user-mapping string into a dictionary."""
    user_service_account_mapping = collections.OrderedDict()
    mapping_str_list = user_service_account_mapping_string.split(',')
    for mapping_str in mapping_str_list:
        mapping = mapping_str.split(':')
        if len(mapping) != 2:
            raise exceptions.ArgumentError('Invalid Secure Multi-Tenancy User Mapping.')
        user_service_account_mapping[mapping[0]] = mapping[1]
    return user_service_account_mapping