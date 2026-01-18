from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def abort_detaching(self, session):
    """Roll back volume status to 'in-use'"""
    body = {'os-roll_detaching': None}
    self._action(session, body)