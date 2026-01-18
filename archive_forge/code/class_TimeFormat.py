import calendar
from datetime import date, datetime, time
from email.utils import format_datetime as format_datetime_rfc5322
from django.utils.dates import (
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import (
from django.utils.translation import gettext as _
class TimeFormat(Formatter):

    def __init__(self, obj):
        self.data = obj
        self.timezone = None
        if isinstance(obj, datetime):
            if is_naive(obj):
                timezone = get_default_timezone()
            else:
                timezone = obj.tzinfo
            if not _datetime_ambiguous_or_imaginary(obj, timezone):
                self.timezone = timezone

    def a(self):
        """'a.m.' or 'p.m.'"""
        if self.data.hour > 11:
            return _('p.m.')
        return _('a.m.')

    def A(self):
        """'AM' or 'PM'"""
        if self.data.hour > 11:
            return _('PM')
        return _('AM')

    def e(self):
        """
        Timezone name.

        If timezone information is not available, return an empty string.
        """
        if not self.timezone:
            return ''
        try:
            if getattr(self.data, 'tzinfo', None):
                return self.data.tzname() or ''
        except NotImplementedError:
            pass
        return ''

    def f(self):
        """
        Time, in 12-hour hours and minutes, with minutes left off if they're
        zero.
        Examples: '1', '1:30', '2:05', '2'
        Proprietary extension.
        """
        hour = self.data.hour % 12 or 12
        minute = self.data.minute
        return '%d:%02d' % (hour, minute) if minute else hour

    def g(self):
        """Hour, 12-hour format without leading zeros; i.e. '1' to '12'"""
        return self.data.hour % 12 or 12

    def G(self):
        """Hour, 24-hour format without leading zeros; i.e. '0' to '23'"""
        return self.data.hour

    def h(self):
        """Hour, 12-hour format; i.e. '01' to '12'"""
        return '%02d' % (self.data.hour % 12 or 12)

    def H(self):
        """Hour, 24-hour format; i.e. '00' to '23'"""
        return '%02d' % self.data.hour

    def i(self):
        """Minutes; i.e. '00' to '59'"""
        return '%02d' % self.data.minute

    def O(self):
        """
        Difference to Greenwich time in hours; e.g. '+0200', '-0430'.

        If timezone information is not available, return an empty string.
        """
        if self.timezone is None:
            return ''
        offset = self.timezone.utcoffset(self.data)
        seconds = offset.days * 86400 + offset.seconds
        sign = '-' if seconds < 0 else '+'
        seconds = abs(seconds)
        return '%s%02d%02d' % (sign, seconds // 3600, seconds // 60 % 60)

    def P(self):
        """
        Time, in 12-hour hours, minutes and 'a.m.'/'p.m.', with minutes left off
        if they're zero and the strings 'midnight' and 'noon' if appropriate.
        Examples: '1 a.m.', '1:30 p.m.', 'midnight', 'noon', '12:30 p.m.'
        Proprietary extension.
        """
        if self.data.minute == 0 and self.data.hour == 0:
            return _('midnight')
        if self.data.minute == 0 and self.data.hour == 12:
            return _('noon')
        return '%s %s' % (self.f(), self.a())

    def s(self):
        """Seconds; i.e. '00' to '59'"""
        return '%02d' % self.data.second

    def T(self):
        """
        Time zone of this machine; e.g. 'EST' or 'MDT'.

        If timezone information is not available, return an empty string.
        """
        if self.timezone is None:
            return ''
        return str(self.timezone.tzname(self.data))

    def u(self):
        """Microseconds; i.e. '000000' to '999999'"""
        return '%06d' % self.data.microsecond

    def Z(self):
        """
        Time zone offset in seconds (i.e. '-43200' to '43200'). The offset for
        timezones west of UTC is always negative, and for those east of UTC is
        always positive.

        If timezone information is not available, return an empty string.
        """
        if self.timezone is None:
            return ''
        offset = self.timezone.utcoffset(self.data)
        return offset.days * 86400 + offset.seconds