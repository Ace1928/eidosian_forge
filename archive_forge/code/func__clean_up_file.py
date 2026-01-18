from http.cookies import SimpleCookie
import time
import random
import os
import datetime
import threading
import tempfile
from paste import wsgilib
from paste import request
def _clean_up_file(self, f, exp_time, now):
    t = f.split('-')
    if len(t) != 2:
        return
    t = t[0]
    try:
        sess_time = datetime.datetime(int(t[0:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]), int(t[10:12]), int(t[12:14]))
    except ValueError:
        return
    if sess_time + exp_time < now:
        os.remove(os.path.join(self.session_file_path, f))