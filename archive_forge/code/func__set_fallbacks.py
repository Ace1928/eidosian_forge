from datetime import datetime
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
def _set_fallbacks(self, fallbacks):
    self._secret_fallbacks = fallbacks