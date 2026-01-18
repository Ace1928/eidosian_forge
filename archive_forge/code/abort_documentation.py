from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dns import transaction_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.core import log
Abort transaction.

  This command aborts the transaction and deletes the transaction file.

  ## EXAMPLES

  To abort the transaction, run:

    $ {command} --zone=MANAGED_ZONE
  