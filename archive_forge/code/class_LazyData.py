import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
@dataclass
class LazyData:
    string: str = None
    value: Any = None
    dtype: str = None