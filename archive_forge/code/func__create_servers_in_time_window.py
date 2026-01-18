import datetime
from novaclient.tests.functional import base
def _create_servers_in_time_window(self):
    start = datetime.datetime.now()
    self._create_server()
    self._create_server()
    end = datetime.datetime.now()
    return (start, end)