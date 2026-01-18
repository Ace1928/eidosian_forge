from unittest import mock
from cinderclient.v3 import client as cinderclient
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine.resources.openstack.cinder import volume as os_vol
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class FakeBackupRestore(object):

    def __init__(self, volume_id='vol-123'):
        self.volume_id = volume_id