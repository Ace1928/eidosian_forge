import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def _pick_alternate_flavor(self):
    """Given the flavor picked in the base class setup, this finds the
        opposite flavor to use for a resize test. For example, if m1.nano is
        the flavor, then use m1.micro, but those are only available if Tempest
        is configured. If m1.tiny, then use m1.small.
        """
    flavor_name = self.flavor.name
    if flavor_name == 'm1.nano':
        return 'm1.micro'
    if flavor_name == 'm1.micro':
        return 'm1.nano'
    if flavor_name == 'm1.tiny':
        return 'm1.small'
    if flavor_name == 'm1.small':
        return 'm1.tiny'
    self.fail('Unable to find alternate for flavor: %s' % flavor_name)