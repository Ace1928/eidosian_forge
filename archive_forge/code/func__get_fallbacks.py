from datetime import datetime
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
def _get_fallbacks(self):
    if self._secret_fallbacks is None:
        return settings.SECRET_KEY_FALLBACKS
    return self._secret_fallbacks