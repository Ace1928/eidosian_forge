from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def _parse_event_buffer(readBuffer, nBytes):
    results = []
    while nBytes > 0:
        fni = ctypes.cast(readBuffer, LPFNI)[0]
        ptr = ctypes.addressof(fni) + FILE_NOTIFY_INFORMATION.FileName.offset
        filename = ctypes.string_at(ptr, fni.FileNameLength)
        results.append((fni.Action, filename.decode('utf-16')))
        numToSkip = fni.NextEntryOffset
        if numToSkip <= 0:
            break
        readBuffer = readBuffer[numToSkip:]
        nBytes -= numToSkip
    return results