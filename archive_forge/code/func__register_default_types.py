from __future__ import annotations
import base64
import json
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Callable, TypeVar
def _register_default_types():
    register_type(datetime, 'datetime', datetime.isoformat, datetime.fromisoformat)
    register_type(date, 'date', lambda o: o.isoformat(), lambda o: datetime.fromisoformat(o).date())
    register_type(time, 'time', lambda o: o.isoformat(), time.fromisoformat)
    register_type(Decimal, 'decimal', str, Decimal)
    register_type(uuid.UUID, 'uuid', lambda o: {'hex': o.hex}, lambda o: uuid.UUID(**o))