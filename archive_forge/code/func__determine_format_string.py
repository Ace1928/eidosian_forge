import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _determine_format_string(dates, datetime_format=None):
    """
    Determine the datetime format string based on the averge number
    of days between data points, or if the user passed in kwarg
    datetime_format, use that as an override.
    """
    avg_days_between_points = (dates[-1] - dates[0]) / float(len(dates))
    if datetime_format is not None:
        return datetime_format
    if avg_days_between_points < 0.33:
        if mdates.num2date(dates[-1]).date() != mdates.num2date(dates[0]).date():
            fmtstring = '%b %d, %H:%M'
        else:
            fmtstring = '%H:%M'
    elif mdates.num2date(dates[-1]).date().year != mdates.num2date(dates[0]).date().year:
        fmtstring = '%Y-%b-%d'
    else:
        fmtstring = '%b %d'
    return fmtstring