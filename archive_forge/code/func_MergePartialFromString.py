from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def MergePartialFromString(self, s):
    """Merges in data from the string 's'.
    Does not enforce required fields are set."""
    try:
        self._CMergeFromString(s)
    except (NotImplementedError, AttributeError):
        a = array.array('B')
        a.fromstring(s)
        d = Decoder(a, 0, len(a))
        self.TryMerge(d)