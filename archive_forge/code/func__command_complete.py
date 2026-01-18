import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _command_complete(self, name, tag):
    logout = name == 'LOGOUT'
    if not logout:
        self._check_bye()
    try:
        typ, data = self._get_tagged_response(tag, expect_bye=logout)
    except self.abort as val:
        raise self.abort('command: %s => %s' % (name, val))
    except self.error as val:
        raise self.error('command: %s => %s' % (name, val))
    if not logout:
        self._check_bye()
    if typ == 'BAD':
        raise self.error('%s command error: %s %s' % (name, typ, data))
    return (typ, data)