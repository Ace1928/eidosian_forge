import time
import logging
import datetime
import functools
from pyzor.engines.common import *
@safe_call
def _get_new_connection(self):
    if '/' in self.host:
        return redis.StrictRedis(unix_socket_path=self.host, db=int(self.db_name), password=self.passwd)
    return redis.StrictRedis(host=self.host, port=int(self.port), db=int(self.db_name), password=self.passwd)