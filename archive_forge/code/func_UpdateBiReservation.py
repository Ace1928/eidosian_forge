from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def UpdateBiReservation(client, reference, reservation_size: str):
    """Updates a BI reservation with the given reservation reference.

  Arguments:
    client: The client used to make the request.
    reference: Reservation to update.
    reservation_size: size of reservation in GBs. It may only contain digits,
      optionally followed by 'G', 'g', 'GB, 'gb', 'gB', or 'Gb'.

  Returns:
    Reservation object that was updated.
  Raises:
    ValueError: if reservation_size is malformed.
  """
    if reservation_size.upper().endswith('GB') and reservation_size[:-2].isdigit():
        reservation_digits = reservation_size[:-2]
    elif reservation_size.upper().endswith('G') and reservation_size[:-1].isdigit():
        reservation_digits = reservation_size[:-1]
    elif reservation_size.isdigit():
        reservation_digits = reservation_size
    else:
        raise ValueError('Invalid reservation size. The unit for BI reservations\n    is GB. The specified reservation size may only contain digits, optionally\n    followed by G, g, GB, gb, gB, or Gb.')
    reservation_size = int(reservation_digits) * 1024 * 1024 * 1024
    bi_reservation = {}
    update_mask = ''
    bi_reservation['size'] = reservation_size
    update_mask += 'size,'
    return client.projects().locations().updateBiReservation(name=reference.path(), updateMask=update_mask, body=bi_reservation).execute()