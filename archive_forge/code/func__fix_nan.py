import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _fix_nan(self, context):
    """Decapitate the payload of a NaN to fit the context"""
    payload = self._int
    max_payload_len = context.prec - context.clamp
    if len(payload) > max_payload_len:
        payload = payload[len(payload) - max_payload_len:].lstrip('0')
        return _dec_from_triple(self._sign, payload, self._exp, True)
    return Decimal(self)