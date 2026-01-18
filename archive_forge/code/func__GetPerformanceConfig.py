from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _GetPerformanceConfig(self, args):
    """Returns the performance config with dump parallel level.

    Args:
      args: argparse.Namespace, the arguments that this command was invoked
        with.
    """
    performance_config_obj = self.messages.PerformanceConfig
    return performance_config_obj(dumpParallelLevel=performance_config_obj.DumpParallelLevelValueValuesEnum.lookup_by_name(args.dump_parallel_level))