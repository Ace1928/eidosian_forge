from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def BuildUpdateMask(self, crypto_key, clear_crypto_key):
    """Builds an update mask for updating a channel.

    Args:
      crypto_key: bool, whether to update the crypto key.
      clear_crypto_key: bool, whether to clear the crypto key.

    Returns:
      The update mask as a string.

    Raises:
      NoFieldsSpecifiedError: No fields are being updated.
    """
    update_mask = []
    if crypto_key:
        update_mask.append('cryptoKeyName')
    if clear_crypto_key:
        update_mask.append('cryptoKeyName')
    if not update_mask:
        raise NoFieldsSpecifiedError('Must specify at least one field to update.')
    return ','.join(update_mask)