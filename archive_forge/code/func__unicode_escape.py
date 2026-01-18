from __future__ import annotations
import re
from collections.abc import Mapping
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from typing import Collection
from tomlkit._compat import decode
def _unicode_escape(seq: str) -> str:
    return ''.join((f'\\u{ord(c):04x}' for c in seq))