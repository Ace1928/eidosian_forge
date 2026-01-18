import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _calculate_load_amount(self, read_size):
    """This calculates how many bytes need to be added to the buffer.

        When a consumer read's ``x`` from the buffer, there are two cases to
        satisfy:

            1. Enough data in the buffer to return the requested amount
            2. Not enough data

        This function uses the amount of unread bytes in the buffer and
        determines how much the Encoder has to load before it can return the
        requested amount of bytes.

        :param int read_size: the number of bytes the consumer requests
        :returns: int -- the number of bytes that must be loaded into the
            buffer before the read can be satisfied. This will be strictly
            non-negative
        """
    amount = read_size - total_len(self._buffer)
    return amount if amount > 0 else 0