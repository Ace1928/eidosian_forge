import re
class DecimalFormatter(Formatter):
    """lets you specify how to build a decimal.

    A future NumberFormatter class will take Microsoft-style patterns
    instead - "$#,##0.00" is WAY easier than this."""

    def __init__(self, places=2, decimalSep='.', thousandSep=None, prefix=None, suffix=None):
        if places == 'auto':
            self.calcPlaces = self._calcPlaces
        else:
            self.places = places
        self.dot = decimalSep
        self.comma = thousandSep
        self.prefix = prefix
        self.suffix = suffix

    def _calcPlaces(self, V):
        """called with the full set of values to be formatted so we can calculate places"""
        self.places = max([len(_tz_re.sub('', _ld_re.sub('', str(v)))) for v in V])

    def format(self, num):
        sign = num < 0
        if sign:
            num = -num
        places, sep = (self.places, self.dot)
        strip = places <= 0
        if places and strip:
            places = -places
        strInt = ('%.' + str(places) + 'f') % num
        if places:
            strInt, strFrac = strInt.split('.')
            strFrac = sep + strFrac
            if strip:
                while strFrac and strFrac[-1] in ['0', sep]:
                    strFrac = strFrac[:-1]
        else:
            strFrac = ''
        if self.comma is not None:
            strNew = ''
            while strInt:
                left, right = (strInt[0:-3], strInt[-3:])
                if left == '':
                    strNew = right + strNew
                else:
                    strNew = self.comma + right + strNew
                strInt = left
            strInt = strNew
        strBody = strInt + strFrac
        if sign:
            strBody = '-' + strBody
        if self.prefix:
            strBody = self.prefix + strBody
        if self.suffix:
            strBody = strBody + self.suffix
        return strBody

    def __repr__(self):
        return '%s(places=%d, decimalSep=%s, thousandSep=%s, prefix=%s, suffix=%s)' % (self.__class__.__name__, self.places, repr(self.dot), repr(self.comma), repr(self.prefix), repr(self.suffix))