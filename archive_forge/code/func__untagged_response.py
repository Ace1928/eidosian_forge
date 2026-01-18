import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _untagged_response(self, typ, dat, name):
    if typ == 'NO':
        return (typ, dat)
    if not name in self.untagged_responses:
        return (typ, [None])
    data = self.untagged_responses.pop(name)
    if __debug__:
        if self.debug >= 5:
            self._mesg('untagged_responses[%s] => %s' % (name, data))
    return (typ, data)