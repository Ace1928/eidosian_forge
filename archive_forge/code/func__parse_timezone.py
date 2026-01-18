import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Optional, Type, Union
from . import errors
def _parse_timezone(value: Optional[str], error: Type[Exception]) -> Union[None, int, timezone]:
    if value == 'Z':
        return timezone.utc
    elif value is not None:
        offset_mins = int(value[-2:]) if len(value) > 3 else 0
        offset = 60 * int(value[1:3]) + offset_mins
        if value[0] == '-':
            offset = -offset
        try:
            return timezone(timedelta(minutes=offset))
        except ValueError:
            raise error()
    else:
        return None