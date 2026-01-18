from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ShareNetworkWithNova(share_network.ManilaShareNetwork):

    def is_using_neutron(self):
        return False