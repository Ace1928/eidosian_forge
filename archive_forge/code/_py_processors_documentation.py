from __future__ import annotations
import datetime
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from datetime import time as time_cls
from decimal import Decimal
import typing
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
defines generic type conversion functions, as used in bind and result
processors.

They all share one common characteristic: None is passed through unchanged.

