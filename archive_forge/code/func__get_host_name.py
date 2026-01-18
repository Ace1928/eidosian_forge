from troveclient import base
from troveclient import common
@staticmethod
def _get_host_name(host):
    try:
        if host.name:
            return host.name
    except AttributeError:
        return host