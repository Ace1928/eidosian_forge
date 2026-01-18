from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class TimeLocale(VegaLiteSchema):
    """TimeLocale schema wrapper
    Locale definition for formatting dates and times.

    Parameters
    ----------

    date : str
        The date (%x) format specifier (e.g., "%m/%d/%Y").
    dateTime : str
        The date and time (%c) format specifier (e.g., "%a %b %e %X %Y").
    days : Sequence[str], :class:`Vector7string`
        The full names of the weekdays, starting with Sunday.
    months : Sequence[str], :class:`Vector12string`
        The full names of the months (starting with January).
    periods : Sequence[str], :class:`Vector2string`
        The A.M. and P.M. equivalents (e.g., ["AM", "PM"]).
    shortDays : Sequence[str], :class:`Vector7string`
        The abbreviated names of the weekdays, starting with Sunday.
    shortMonths : Sequence[str], :class:`Vector12string`
        The abbreviated names of the months (starting with January).
    time : str
        The time (%X) format specifier (e.g., "%H:%M:%S").
    """
    _schema = {'$ref': '#/definitions/TimeLocale'}

    def __init__(self, date: Union[str, UndefinedType]=Undefined, dateTime: Union[str, UndefinedType]=Undefined, days: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, months: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, periods: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, shortDays: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, shortMonths: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, time: Union[str, UndefinedType]=Undefined, **kwds):
        super(TimeLocale, self).__init__(date=date, dateTime=dateTime, days=days, months=months, periods=periods, shortDays=shortDays, shortMonths=shortMonths, time=time, **kwds)