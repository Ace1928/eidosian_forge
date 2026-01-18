from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def WaitForHubOp(self, poller, op, message=None, warnings=True, **kwargs):
    """Helper wrapping waiter.WaitFor() with additional warning handling."""
    op_ref = self.hubclient.OperationRef(op)
    result = waiter.WaitFor(poller, op_ref, message=message, **kwargs)
    if warnings:
        final_op = poller.Poll(op_ref)
        metadata_dict = encoding.MessageToPyValue(final_op.metadata)
        if 'statusDetail' in metadata_dict:
            log.warning(metadata_dict['statusDetail'])
    return result