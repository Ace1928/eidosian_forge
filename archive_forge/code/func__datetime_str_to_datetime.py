import datetime
from functools import partial
import logging
def _datetime_str_to_datetime(datetime_str, format='%Y-%m-%dT%H:%M:%S'):
    """ Returns the datetime object for a datetime string in the specified
    format (default ISO format).

    Raises a ValueError if datetime_str does not match the format.
    """
    if not datetime_str:
        return None
    return datetime.datetime.strptime(datetime_str, format)