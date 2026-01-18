from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
class _BaseWorker(worker.BaseWorker):

    def reset(self):
        pass

    def stop(self):
        pass

    def wait(self):
        pass