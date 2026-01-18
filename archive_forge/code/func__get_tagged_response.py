import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _get_tagged_response(self, tag, expect_bye=False):
    while 1:
        result = self.tagged_commands[tag]
        if result is not None:
            del self.tagged_commands[tag]
            return result
        if expect_bye:
            typ = 'BYE'
            bye = self.untagged_responses.pop(typ, None)
            if bye is not None:
                return (typ, bye)
        self._check_bye()
        try:
            self._get_response()
        except self.abort as val:
            if __debug__:
                if self.debug >= 1:
                    self.print_log()
            raise