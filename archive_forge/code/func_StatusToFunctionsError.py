from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def StatusToFunctionsError(status, error_message=None):
    """Convert a google.rpc.Status (used for LRO errors) into a FunctionsError."""
    if error_message:
        return FunctionsError(error_message)
    return FunctionsError(status.message)