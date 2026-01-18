from unittest import mock
from magnumclient import exceptions as mc_exc
from heat.engine.clients.os import magnum as mc
from heat.tests import common
from heat.tests import utils
class fake_cluster_template(object):

    def __init__(self, id=None, name=None):
        self.uuid = id
        self.name = name