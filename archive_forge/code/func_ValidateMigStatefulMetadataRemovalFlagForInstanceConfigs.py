from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateMigStatefulMetadataRemovalFlagForInstanceConfigs(entries_to_remove, entries_to_update):
    remove_stateful_metadata_set = set(entries_to_remove or [])
    update_stateful_metadata_set = set(entries_to_update.keys())
    keys_intersection = remove_stateful_metadata_set.intersection(update_stateful_metadata_set)
    if keys_intersection:
        raise exceptions.InvalidArgumentException(parameter_name='--remove-stateful-metadata', message='the same metadata key(s) `{0}` cannot be updated and removed in one command call'.format(', '.join(keys_intersection)))