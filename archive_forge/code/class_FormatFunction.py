import datetime
from typing import Protocol, Tuple, Type, Union
class FormatFunction(Protocol):

    def __call__(self, sequence: str) -> Union[int, float, bool, datetime.date, datetime.time, datetime.datetime]:
        ...