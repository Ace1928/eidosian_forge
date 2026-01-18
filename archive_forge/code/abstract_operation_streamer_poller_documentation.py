from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
Checks if need to init a new output streamer.

    Checks if need to init a new output streamer.
    Remote may fail; switch to new output uri.
    Invalidate the streamer instance and init a new one if necessary.

    Args:
      poll_result: Poll result returned from Poll.
    