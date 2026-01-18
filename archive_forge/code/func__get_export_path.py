import os
from os_win import utilsfactory
from os_brick.initiator.windows import base as win_conn_base
from os_brick.remotefs import windows_remotefs as remotefs
from os_brick import utils
def _get_export_path(self, connection_properties):
    return connection_properties['export'].replace('/', '\\')