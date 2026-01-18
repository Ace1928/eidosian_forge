from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from ..util.deprecation import deprecated
from ..util.strings import format_docstring
from ..util.warnings import warn
from .tickers import Ticker
class DatetimeTickFormatter(TickFormatter):
    """ A ``TickFormatter`` for displaying datetime values nicely across a
    range of scales.

    ``DatetimeTickFormatter`` has the following properties (listed together
    with their default values) that can be used to control the formatting
    of axis ticks at different scales:

    .. code-block:: python

        {defaults}

    Each scale property can be set to format or list of formats to use for
    formatting datetime tick values that fall in in that "time scale".
    By default, only the first format string passed for each time scale
    will be used. By default, all leading zeros are stripped away from
    the formatted labels.

    This list of supported `strftime`_ formats is reproduced below.

    %a
        The abbreviated name of the day of the week according to the
        current locale.

    %A
        The full name of the day of the week according to the current
        locale.

    %b
        The abbreviated month name according to the current locale.

    %B
        The full month name according to the current locale.

    %c
        The preferred date and time representation for the current
        locale.

    %C
        The century number (year/100) as a 2-digit integer.

    %d
        The day of the month as a decimal number (range 01 to 31).

    %D
        Equivalent to **%m/%d/%y**.  (Americans should note that in many
        other countries **%d/%m/%y** is rather common. This means that in
        international context this format is ambiguous and should not
        be used.)

    %e
        Like %d, the day of the month as a decimal number, but a
        leading zero is replaced by a space.

    %f
        Microsecond as a decimal number, zero-padded on the left (range
        000000-999999). This is an extension to the set of directives
        available to `timezone`_.

    %F
        Equivalent to **%Y-%m-%d** (the ISO 8601 date format).

    %G
        The ISO 8601 week-based year with century as a decimal number.
        The 4-digit year corresponding to the ISO week number (see %V).
        This has the same format and value as %Y, except that if the
        ISO week number belongs to the previous or next year, that year
        is used instead.

    %g
        Like **%G**, but without century, that is, with a 2-digit year (00-99).

    %h
        Equivalent to **%b**.

    %H
        The hour as a decimal number using a 24-hour clock (range 00
        to 23).

    %I
        The hour as a decimal number using a 12-hour clock (range 01
        to 12).

    %j
        The day of the year as a decimal number (range 001 to 366).

    %k
        The hour (24-hour clock) as a decimal number (range 0 to 23).
        Single digits are preceded by a blank. See also **%H**.

    %l
        The hour (12-hour clock) as a decimal number (range 1 to 12).
        Single digits are preceded by a blank. See also **%I**.

    %m
        The month as a decimal number (range 01 to 12).

    %M
        The minute as a decimal number (range 00 to 59).

    %n
        A newline character. Bokeh text does not currently support
        newline characters.

    %N
        Nanosecond as a decimal number, zero-padded on the left (range
        000000000-999999999). Supports a padding width specifier, i.e.
        %3N displays 3 leftmost digits. However, this is only accurate
        to the millisecond level of precision due to limitations of
        `timezone`_.

    %p
        Either "AM" or "PM" according to the given time value, or the
        corresponding strings for the current locale.  Noon is treated
        as "PM" and midnight as "AM".

    %P
        Like %p but in lowercase: "am" or "pm" or a corresponding
        string for the current locale.

    %r
        The time in a.m. or p.m. notation.  In the POSIX locale this
        is equivalent to **%I:%M:%S %p**.

    %R
        The time in 24-hour notation (**%H:%M**). For a version including
        the seconds, see **%T** below.

    %s
        The number of seconds since the Epoch, 1970-01-01 00:00:00
        +0000 (UTC).

    %S
        The second as a decimal number (range 00 to 60).  (The range
        is up to 60 to allow for occasional leap seconds.)

    %t
        A tab character. Bokeh text does not currently support tab
        characters.

    %T
        The time in 24-hour notation (**%H:%M:%S**).

    %u
        The day of the week as a decimal, range 1 to 7, Monday being 1.
        See also %w.

    %U
        The week number of the current year as a decimal number, range
        00 to 53, starting with the first Sunday as the first day of
        week 01.  See also **%V** and **%W**.

    %V
        The ISO 8601 week number (see NOTES) of the current year as a
        decimal number, range 01 to 53, where week 1 is the first week
        that has at least 4 days in the new year.  See also %U and %W.

    %w
        The day of the week as a decimal, range 0 to 6, Sunday being 0.
        See also %u.

    %W
        The week number of the current year as a decimal number, range
        00 to 53, starting with the first Monday as the first day of
        week 01.

    %x
        The preferred date representation for the current locale
        without the time.

    %X
        The preferred time representation for the current locale
        without the date.

    %y
        The year as a decimal number without a century (range 00 to 99).

    %Y
        The year as a decimal number including the century.

    %z
        The +hhmm or -hhmm numeric timezone (that is, the hour and
        minute offset from UTC).

    %Z
        The timezone name or abbreviation.

    %%
        A literal '%' character.

    .. warning::
        The client library BokehJS uses the `timezone`_ library to
        format datetimes. The inclusion of the list below is based on the
        claim that `timezone`_ makes to support "the full compliment
        of GNU date format specifiers." However, this claim has not
        been tested exhaustively against this list. If you find formats
        that do not function as expected, please submit a `github issue`_,
        so that the documentation can be updated appropriately.

    .. _strftime: http://man7.org/linux/man-pages/man3/strftime.3.html
    .. _timezone: http://bigeasy.github.io/timezone/
    .. _github issue: https://github.com/bokeh/bokeh/issues

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    microseconds = String(help=_DATETIME_TICK_FORMATTER_HELP('``microseconds``'), default='%fus').accepts(List(String), _deprecated_datetime_list_format)
    milliseconds = String(help=_DATETIME_TICK_FORMATTER_HELP('``milliseconds``'), default='%3Nms').accepts(List(String), _deprecated_datetime_list_format)
    seconds = String(help=_DATETIME_TICK_FORMATTER_HELP('``seconds``'), default='%Ss').accepts(List(String), _deprecated_datetime_list_format)
    minsec = String(help=_DATETIME_TICK_FORMATTER_HELP('``minsec`` (for combined minutes and seconds)'), default=':%M:%S').accepts(List(String), _deprecated_datetime_list_format)
    minutes = String(help=_DATETIME_TICK_FORMATTER_HELP('``minutes``'), default=':%M').accepts(List(String), _deprecated_datetime_list_format)
    hourmin = String(help=_DATETIME_TICK_FORMATTER_HELP('``hourmin`` (for combined hours and minutes)'), default='%H:%M').accepts(List(String), _deprecated_datetime_list_format)
    hours = String(help=_DATETIME_TICK_FORMATTER_HELP('``hours``'), default='%Hh').accepts(List(String), _deprecated_datetime_list_format)
    days = String(help=_DATETIME_TICK_FORMATTER_HELP('``days``'), default='%m/%d').accepts(List(String), _deprecated_datetime_list_format)
    months = String(help=_DATETIME_TICK_FORMATTER_HELP('``months``'), default='%m/%Y').accepts(List(String), _deprecated_datetime_list_format)
    years = String(help=_DATETIME_TICK_FORMATTER_HELP('``years``'), default='%Y').accepts(List(String), _deprecated_datetime_list_format)
    strip_leading_zeros = Either(Bool, Seq(Enum(ResolutionType)), default=False, help='\n    Whether to strip any leading zeros in the formatted ticks.\n\n    Valid values are:\n\n    * ``True`` or ``False`` (default) to set stripping across all resolutions.\n    * A sequence of resolution types, e.g. ``["microseconds", "milliseconds"]``, to enable\n      scale-dependent stripping of leading zeros.\n    ')
    context = Nullable(Either(String, Instance('bokeh.models.formatters.DatetimeTickFormatter')), default=None, help='\n    A format for adding context to the tick or ticks specified by ``context_which``.\n    Valid values are:\n\n    * None, no context is added\n    * A standard :class:`~bokeh.models.DatetimeTickFormatter` format string, the single format is\n      used across all scales\n    * Another :class:`~bokeh.models.DatetimeTickFormatter` instance, to have scale-dependent\n      context\n    ')
    context_which = Enum(ContextWhich, default='start', help='\n    Which tick or ticks to add a formatted context string to. Valid values are:\n    `"start"`, `"end"`, `"center"`, and  `"all"`.\n    ')
    context_location = Enum(LocationType, default='below', help='\n    Relative to the tick label text baseline, where the context should be\n    rendered. Valid values are: `"below"`, `"above"`, `"left"`, and `"right"`.\n    ')