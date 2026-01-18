from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
class TestBaseWorker(base.BaseTestCase):

    def setUp(self):
        super(TestBaseWorker, self).setUp()
        self._reg = mock.Mock()
        self.useFixture(fixture.CallbackRegistryFixture(callback_manager=self._reg))

    def test_worker_process_count(self):
        self.assertEqual(9, _BaseWorker(worker_process_count=9).worker_process_count)

    def test_start_callback_event(self):
        base_worker = _BaseWorker()
        base_worker.start()
        self._reg.publish.assert_called_once_with(resources.PROCESS, events.AFTER_INIT, base_worker.start, payload=mock.ANY)

    def test_proctitle_default(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start()
            self.assertRegex(spt.call_args[0][0], '^neutron-server: _ProcWorker \\(.*python.*\\)$')

    def test_proctitle_custom_desc(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(desc='fancy title')
            self.assertRegex(spt.call_args[0][0], '^neutron-server: fancy title \\(.*python.*\\)$')

    def test_proctitle_custom_name(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(name='tardis')
            self.assertRegex(spt.call_args[0][0], '^tardis: _ProcWorker \\(.*python.*\\)$')

    def test_proctitle_empty(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(desc='')
            self.assertRegex(spt.call_args[0][0], '^neutron-server: _ProcWorker \\(.*python.*\\)$')

    def test_proctitle_nonstring(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(desc=2)
            self.assertRegex(spt.call_args[0][0], '^neutron-server: 2 \\(.*python.*\\)$')

    def test_proctitle_both_empty(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(name='', desc='')
            self.assertRegex(spt.call_args[0][0], '^: _ProcWorker \\(.*python.*\\)$')

    def test_proctitle_name_none(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker().start(name=None)
            self.assertRegex(spt.call_args[0][0], '^None: _ProcWorker \\(.*python.*\\)$')

    def test_proctitle_off(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker(set_proctitle='off').start()
            self.assertIsNone(spt.call_args)

    def test_proctitle_same_process(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _BaseWorker().start()
            self.assertIsNone(spt.call_args)

    def test_setproctitle_on(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker(set_proctitle='on').start(name='foo', desc='bar')
            self.assertRegex(spt.call_args[0][0], '^foo: bar \\(.*python.*\\)$')

    def test_setproctitle_off(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker(set_proctitle='off').start(name='foo', desc='bar')
            self.assertIsNone(spt.call_args)

    def test_setproctitle_brief(self):
        with mock.patch('setproctitle.setproctitle') as spt:
            _ProcWorker(set_proctitle='brief').start(name='foo', desc='bar')
            self.assertEqual(spt.call_args[0][0], 'foo: bar')