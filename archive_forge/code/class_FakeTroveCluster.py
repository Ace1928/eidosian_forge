import copy
from unittest import mock
from troveclient import exceptions as troveexc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine.resources.openstack.trove import cluster
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeTroveCluster(object):

    def __init__(self, status='ACTIVE'):
        self.name = 'cluster'
        self.id = '1189aa64-a471-4aa3-876a-9eb7d84089da'
        self.ip = ['10.0.0.1']
        self.instances = [{'id': '416b0b16-ba55-4302-bbd3-ff566032e1c1', 'status': status}, {'id': '965ef811-7c1d-47fc-89f2-a89dfdd23ef2', 'status': status}, {'id': '3642f41c-e8ad-4164-a089-3891bf7f2d2b', 'status': status}]

    def delete(self):
        pass