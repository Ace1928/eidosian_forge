from __future__ import annotations
import logging # isort:skip
import base64
import calendar
import codecs
import datetime as dt
import hashlib
import hmac
import json
import time
import zlib
from typing import TYPE_CHECKING, Any
from ..core.types import ID
from ..settings import settings
from .warnings import warn
def _get_sysrandom() -> tuple[Any, bool]:
    import random
    try:
        sysrandom = random.SystemRandom()
        using_sysrandom = True
        return (sysrandom, using_sysrandom)
    except NotImplementedError:
        warn('A secure pseudo-random number generator is not available on your system. Falling back to Mersenne Twister.')
        if settings.secret_key() is None:
            warn('A secure pseudo-random number generator is not available and no BOKEH_SECRET_KEY has been set. Setting a secret key will mitigate the lack of a secure generator.')
        using_sysrandom = False
        return (random, using_sysrandom)