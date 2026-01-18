from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def abort_introspection(self, introspection, ignore_missing=True):
    """Abort an introspection.

        Note that the introspection is not aborted immediately, you may use
        `wait_for_introspection` with `ignore_error=True`.

        :param introspection: The value can be the name or ID of an
            introspection (matching bare metal node name or ID) or
            an :class:`~.introspection.Introspection` instance.
        :param bool ignore_missing: When set to ``False``, an exception
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the introspection could not be found. When set to ``True``, no
            exception will be raised when attempting to abort a non-existent
            introspection.
        :returns: nothing
        """
    res = self._get_resource(_introspect.Introspection, introspection)
    try:
        res.abort(self)
    except exceptions.ResourceNotFound:
        if not ignore_missing:
            raise