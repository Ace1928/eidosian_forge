import datetime
from functools import partial
import logging
def _datetime_to_datetime_str(datetime_obj, format='%Y-%m-%dT%H:%M:%S'):
    """ Returns a string representation for a datetime object in the specified
    format (default ISO format).
    """
    if datetime_obj is None:
        return ''
    return datetime.date.strftime(datetime_obj, format)