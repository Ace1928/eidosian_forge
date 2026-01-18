import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
class WaitResult(collections.namedtuple('WaitResult', ['success', 'failure', 'timeout'])):
    """A named tuple representing a result of waiting for several nodes.

    Each component is a list of :class:`~openstack.baremetal.v1.node.Node`
    objects:

    :ivar ~.success: a list of :class:`~openstack.baremetal.v1.node.Node`
        objects that reached the state.
    :ivar ~.timeout: a list of :class:`~openstack.baremetal.v1.node.Node`
        objects that reached timeout.
    :ivar ~.failure: a list of :class:`~openstack.baremetal.v1.node.Node`
        objects that hit a failure.
    """
    __slots__ = ()