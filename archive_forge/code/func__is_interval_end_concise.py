import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def _is_interval_end_concise(cls, endtuple):
    if type(endtuple) is TimeTuple:
        return True
    if type(endtuple) is DatetimeTuple:
        enddatetuple = endtuple.date
    else:
        enddatetuple = endtuple
    if enddatetuple.YYYY is None:
        return True
    return False