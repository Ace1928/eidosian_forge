from datetime import datetime
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
def _make_token_with_timestamp(self, user, timestamp, secret):
    ts_b36 = int_to_base36(timestamp)
    hash_string = salted_hmac(self.key_salt, self._make_hash_value(user, timestamp), secret=secret, algorithm=self.algorithm).hexdigest()[::2]
    return '%s-%s' % (ts_b36, hash_string)