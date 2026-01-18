import datetime
import os
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def _update_forward_compatibility_date_number(date_to_override=None):
    """Update the base date to compare in forward_compatible function."""
    global _FORWARD_COMPATIBILITY_DATE_NUMBER
    if date_to_override:
        date = date_to_override
    else:
        date = _FORWARD_COMPATIBILITY_HORIZON
        delta_days = os.getenv(_FORWARD_COMPATIBILITY_DELTA_DAYS_VAR_NAME)
        if delta_days:
            date += datetime.timedelta(days=int(delta_days))
    if date < _FORWARD_COMPATIBILITY_HORIZON:
        logging.warning('Trying to set the forward compatibility date to the past date %s. This will be ignored by TensorFlow.' % date)
        return
    _FORWARD_COMPATIBILITY_DATE_NUMBER = _date_to_date_number(date.year, date.month, date.day)