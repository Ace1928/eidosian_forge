from datetime import timedelta, tzinfo
import time
def _Utc():
    """
    Helper function for unpickling a Utc object.
    """
    return UTC