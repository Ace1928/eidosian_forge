from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class TPUInUnusableState(exceptions.Error):
    """Error when the TPU is in an unusable state."""

    def __init__(self, state):
        super(TPUInUnusableState, self).__init__('This TPU has state "{}", so it cannot be currently connected to.'.format(state))