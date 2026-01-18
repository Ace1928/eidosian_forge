import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
Wait for a baremetal node to have no lock.

        DEPRECATED, use ``wait_for_node_reservation`` on the `baremetal` proxy.

        :raises: :class:`~openstack.exceptions.SDKException` upon client
            failure.
        :returns: None
        