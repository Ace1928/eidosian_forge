import datetime
import numpy as np
import shapely.geometry as sgeom
from .. import crs as ccrs
from . import ShapelyFeature
def _julian_day(date):
    """
    Calculate the Julian day from an input datetime.

    Parameters
    ----------
    date
        A UTC datetime object.

    Note
    ----
    Algorithm implemented following equations from Chapter 3 (Algorithm 14):
    Vallado, David 'Fundamentals of Astrodynamics and Applications', (2007)

    Julian day epoch is: noon on January 1, 4713 BC (proleptic Julian)
                         noon on November 24, 4714 BC (proleptic Gregorian)

    """
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    if month < 3:
        month += 12
        year -= 1
    B = 2 - year // 100 + year // 100 // 4
    C = ((second / 60 + minute) / 60 + hour) / 24
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5 + C
    return JD