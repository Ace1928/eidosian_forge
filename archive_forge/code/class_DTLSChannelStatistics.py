from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
@attrs.frozen
class DTLSChannelStatistics:
    """Currently this has only one attribute:

    - ``incoming_packets_dropped_in_trio`` (``int``): Gives a count of the number of
      incoming packets from this peer that Trio successfully received from the
      network, but then got dropped because the internal channel buffer was full. If
      this is non-zero, then you might want to call ``receive`` more often, or use a
      larger ``incoming_packets_buffer``, or just not worry about it because your
      UDP-based protocol should be able to handle the occasional lost packet, right?

    """
    incoming_packets_dropped_in_trio: int