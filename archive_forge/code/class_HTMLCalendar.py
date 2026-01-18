import sys
import datetime
import locale as _locale
from itertools import repeat
class HTMLCalendar(Calendar):
    """
    This calendar returns complete HTML pages.
    """
    cssclasses = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    cssclasses_weekday_head = cssclasses
    cssclass_noday = 'noday'
    cssclass_month_head = 'month'
    cssclass_month = 'month'
    cssclass_year_head = 'year'
    cssclass_year = 'year'

    def formatday(self, day, weekday):
        """
        Return a day as a table cell.
        """
        if day == 0:
            return '<td class="%s">&nbsp;</td>' % self.cssclass_noday
        else:
            return '<td class="%s">%d</td>' % (self.cssclasses[weekday], day)

    def formatweek(self, theweek):
        """
        Return a complete week as a table row.
        """
        s = ''.join((self.formatday(d, wd) for d, wd in theweek))
        return '<tr>%s</tr>' % s

    def formatweekday(self, day):
        """
        Return a weekday name as a table header.
        """
        return '<th class="%s">%s</th>' % (self.cssclasses_weekday_head[day], day_abbr[day])

    def formatweekheader(self):
        """
        Return a header for a week as a table row.
        """
        s = ''.join((self.formatweekday(i) for i in self.iterweekdays()))
        return '<tr>%s</tr>' % s

    def formatmonthname(self, theyear, themonth, withyear=True):
        """
        Return a month name as a table row.
        """
        if withyear:
            s = '%s %s' % (month_name[themonth], theyear)
        else:
            s = '%s' % month_name[themonth]
        return '<tr><th colspan="7" class="%s">%s</th></tr>' % (self.cssclass_month_head, s)

    def formatmonth(self, theyear, themonth, withyear=True):
        """
        Return a formatted month as a table.
        """
        v = []
        a = v.append
        a('<table border="0" cellpadding="0" cellspacing="0" class="%s">' % self.cssclass_month)
        a('\n')
        a(self.formatmonthname(theyear, themonth, withyear=withyear))
        a('\n')
        a(self.formatweekheader())
        a('\n')
        for week in self.monthdays2calendar(theyear, themonth):
            a(self.formatweek(week))
            a('\n')
        a('</table>')
        a('\n')
        return ''.join(v)

    def formatyear(self, theyear, width=3):
        """
        Return a formatted year as a table of tables.
        """
        v = []
        a = v.append
        width = max(width, 1)
        a('<table border="0" cellpadding="0" cellspacing="0" class="%s">' % self.cssclass_year)
        a('\n')
        a('<tr><th colspan="%d" class="%s">%s</th></tr>' % (width, self.cssclass_year_head, theyear))
        for i in range(January, January + 12, width):
            months = range(i, min(i + width, 13))
            a('<tr>')
            for m in months:
                a('<td>')
                a(self.formatmonth(theyear, m, withyear=False))
                a('</td>')
            a('</tr>')
        a('</table>')
        return ''.join(v)

    def formatyearpage(self, theyear, width=3, css='calendar.css', encoding=None):
        """
        Return a formatted year as a complete HTML page.
        """
        if encoding is None:
            encoding = sys.getdefaultencoding()
        v = []
        a = v.append
        a('<?xml version="1.0" encoding="%s"?>\n' % encoding)
        a('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n')
        a('<html>\n')
        a('<head>\n')
        a('<meta http-equiv="Content-Type" content="text/html; charset=%s" />\n' % encoding)
        if css is not None:
            a('<link rel="stylesheet" type="text/css" href="%s" />\n' % css)
        a('<title>Calendar for %d</title>\n' % theyear)
        a('</head>\n')
        a('<body>\n')
        a(self.formatyear(theyear, width))
        a('</body>\n')
        a('</html>\n')
        return ''.join(v).encode(encoding, 'xmlcharrefreplace')