from __future__ import annotations
import logging
import re
import sys
import typing as t
from datetime import datetime
from datetime import timezone
def _wsgi_decoding_dance(s: str) -> str:
    return s.encode('latin1').decode(errors='replace')