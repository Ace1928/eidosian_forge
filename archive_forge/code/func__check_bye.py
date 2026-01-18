import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _check_bye(self):
    bye = self.untagged_responses.get('BYE')
    if bye:
        raise self.abort(bye[-1].decode(self._encoding, 'replace'))