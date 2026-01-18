from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.dataproc.poller import batch_poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.dataproc.batches import (
from googlecloudsdk.core import log
Submits a batch workload.

  Submits a batch workload and streams output if necessary.
  Make sure the parsed argument contains all the necessary arguments before
  calling. It should be fine if the arg parser was passed to
  BatchesCreateRequestFactory's AddArguments function previously.

  Args:
    batch_workload_message: A batch workload message. For example, a SparkBatch
    instance.
    dataproc: An api_lib.dataproc.Dataproc instance.
    args: Parsed arguments.

  Returns:
    Remote return value for a BatchesCreate request.
  