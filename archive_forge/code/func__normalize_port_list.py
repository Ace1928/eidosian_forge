import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def _normalize_port_list(nics):
    ports = []
    for row in nics:
        if isinstance(row, str):
            address = row
            row = {}
        elif 'mac' in row:
            address = row.pop('mac')
        else:
            try:
                address = row.pop('address')
            except KeyError:
                raise TypeError("Either 'address' or 'mac' must be provided for port %s" % row)
        ports.append(dict(row, address=address))
    return ports