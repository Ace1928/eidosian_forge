import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Optional, Type, Union
from . import errors

    Parse a duration int/float/string and return a datetime.timedelta.

    The preferred format for durations in Django is '%d %H:%M:%S.%f'.

    Also supports ISO 8601 representation.
    