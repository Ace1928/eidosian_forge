from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def PrintOperationResultWithWaitEpilogue(operation_ref, result):
    """Prints the operation result with wait epilogue.

  Args:
    operation_ref: Resource reference for the operation
    result: Epiloque string to be printed
  """
    log.status.Print('{}. Use the following command to wait for its completion:\n\ngcloud api-gateway operations wait {}\n'.format(result, operation_ref.RelativeName()))