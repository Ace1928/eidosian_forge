import datetime
import decimal
from .base import Database
class Oracle_datetime(datetime.datetime):
    """
    A datetime object, with an additional class attribute
    to tell oracledb to save the microseconds too.
    """
    input_size = Database.TIMESTAMP

    @classmethod
    def from_datetime(cls, dt):
        return Oracle_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)