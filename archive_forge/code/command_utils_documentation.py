from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
Overrides to get the response from the completed job by job type.

    Args:
      job: api_name_messages.Job.

    Returns:
      the 'response' field of the Operation.
    