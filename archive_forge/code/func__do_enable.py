from manilaclient import api_versions
from manilaclient import base
def _do_enable(self, host, binary, resource_path=RESOURCE_PATH):
    """Enable the service specified by hostname and binary."""
    body = {'host': host, 'binary': binary}
    return self._update('%s/enable' % resource_path, body)