import math
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple
import annotated_types as at
class MyCustomGroupedMetadata(at.GroupedMetadata):

    def __iter__(self) -> Iterator[at.Predicate]:
        yield at.Predicate(lambda x: float(x).is_integer())