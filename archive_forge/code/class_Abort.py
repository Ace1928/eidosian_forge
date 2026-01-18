from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dns import transaction_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.core import log
class Abort(base.Command):
    """Abort transaction.

  This command aborts the transaction and deletes the transaction file.

  ## EXAMPLES

  To abort the transaction, run:

    $ {command} --zone=MANAGED_ZONE
  """

    @staticmethod
    def Args(parser):
        flags.GetZoneArg().AddToParser(parser)

    def Run(self, args):
        if not os.path.isfile(args.transaction_file):
            raise transaction_util.TransactionFileNotFound('Transaction not found at [{0}]'.format(args.transaction_file))
        os.remove(args.transaction_file)
        log.status.Print('Aborted transaction [{0}].'.format(args.transaction_file))