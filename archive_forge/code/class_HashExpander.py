from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
class HashExpander:

    def __init__(self, cronit):
        self.cron = cronit

    def do(self, idx, hash_type='h', hash_id=None, range_end=None, range_begin=None):
        """Return a hashed/random integer given range/hash information"""
        hours_or_minutes = idx in {0, 1}
        if range_end is None:
            range_end = self.cron.RANGES[idx][1]
            if hours_or_minutes:
                range_end += 1
        if range_begin is None:
            range_begin = self.cron.RANGES[idx][0]
        if hash_type == 'r':
            crc = random.randint(0, 4294967295)
        else:
            crc = binascii.crc32(hash_id) & 4294967295
        if not hours_or_minutes:
            return (crc >> idx) % (range_end - range_begin + 1) + range_begin
        return (crc >> idx) % (range_end - range_begin) + range_begin

    def match(self, efl, idx, expr, hash_id=None, **kw):
        return hash_expression_re.match(expr)

    def expand(self, efl, idx, expr, hash_id=None, match='', **kw):
        """Expand a hashed/random expression to its normal representation"""
        if match == '':
            match = self.match(efl, idx, expr, hash_id, **kw)
        if not match:
            return expr
        m = match.groupdict()
        if m['hash_type'] == 'h' and hash_id is None:
            raise CroniterBadCronError('Hashed definitions must include hash_id')
        if m['range_begin'] and m['range_end']:
            if int(m['range_begin']) >= int(m['range_end']):
                raise CroniterBadCronError('Range end must be greater than range begin')
        if m['range_begin'] and m['range_end'] and m['divisor']:
            if int(m['divisor']) == 0:
                raise CroniterBadCronError('Bad expression: {0}'.format(expr))
            return '{0}-{1}/{2}'.format(self.do(idx, hash_type=m['hash_type'], hash_id=hash_id, range_end=int(m['divisor'])) + int(m['range_begin']), int(m['range_end']), int(m['divisor']))
        elif m['range_begin'] and m['range_end']:
            return str(self.do(idx, hash_type=m['hash_type'], hash_id=hash_id, range_end=int(m['range_end']), range_begin=int(m['range_begin'])))
        elif m['divisor']:
            if int(m['divisor']) == 0:
                raise CroniterBadCronError('Bad expression: {0}'.format(expr))
            return '{0}-{1}/{2}'.format(self.do(idx, hash_type=m['hash_type'], hash_id=hash_id, range_end=int(m['divisor'])), self.cron.RANGES[idx][1], int(m['divisor']))
        else:
            return str(self.do(idx, hash_type=m['hash_type'], hash_id=hash_id))