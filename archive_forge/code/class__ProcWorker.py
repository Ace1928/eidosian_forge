from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
class _ProcWorker(_BaseWorker):

    def __init__(self, worker_process_count=1, set_proctitle='on'):
        super(_ProcWorker, self).__init__(worker_process_count, set_proctitle)
        self._my_pid = -1