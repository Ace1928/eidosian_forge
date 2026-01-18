from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def Output(self, e):
    """write self to the encoder 'e'."""
    dbg = []
    if not self.IsInitialized(dbg):
        raise ProtocolBufferEncodeError('\n\t'.join(dbg))
    self.OutputUnchecked(e)
    return