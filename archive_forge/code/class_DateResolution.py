from aniso8601 import compat
class DateResolution(object):
    Year, Month, Week, Weekday, Day, Ordinal = list(compat.range(6))