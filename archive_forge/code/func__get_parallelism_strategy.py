from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _get_parallelism_strategy():
    if task_util.should_use_parallelism():
        return ParallelismStrategy.PARALLEL.value
    return ParallelismStrategy.SEQUENTIAL.value