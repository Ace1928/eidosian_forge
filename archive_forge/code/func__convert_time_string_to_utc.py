import datetime
import time
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
@staticmethod
def _convert_time_string_to_utc(time_string):
    datetime_format = '%Y-%m-%d %H:%M'
    the_time = time_string
    if the_time:
        the_time = datetime.datetime.strptime(the_time, datetime_format)
        is_dst = time.daylight and time.localtime().tm_isdst > 0
        utc_offset = -(time.altzone if is_dst else time.timezone)
        the_time = (the_time - datetime.timedelta(0, utc_offset)).strftime(datetime_format)
    return the_time