from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
import six
def ExecuteTasks(tasks, num_threads=DEFAULT_NUM_THREADS, progress_bar_label=None):
    """Perform the given storage tasks in parallel.

  Factors out common work: logging, setting up parallelism, managing a progress
  bar (if necessary).

  Args:
    tasks: [Operation], To be executed in parallel.
    num_threads: int, The number of threads to use
    progress_bar_label: str, If set, a progress bar will be shown with this
      label. Otherwise, no progress bar is displayed.
  """
    log.debug(progress_bar_label)
    log.debug('Using [%d] threads', num_threads)
    pool = parallel.GetPool(num_threads)
    if progress_bar_label:
        progress_bar = console_io.TickableProgressBar(len(tasks), progress_bar_label)
        callback = progress_bar.Tick
    else:
        progress_bar = console_io.NoOpProgressBar()
        callback = None
    if num_threads == 0:
        with progress_bar:
            for t in tasks:
                t.Execute(callback)
    else:
        with progress_bar, pool:
            pool.Map(lambda task: task.Execute(callback), tasks)