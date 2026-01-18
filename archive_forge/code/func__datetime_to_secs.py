import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
def _datetime_to_secs(utc_time):
    epoch = datetime.datetime(1970, 1, 1)
    time_delta = utc_time - epoch
    return time_delta.days * 86400 + time_delta.seconds