from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def LogCanceledOperation(response, args):
    operation = args.CONCEPTS.operation.Parse()
    log.status.Print('Cancellation in progress for [{}].'.format(operation.Name()))
    return response