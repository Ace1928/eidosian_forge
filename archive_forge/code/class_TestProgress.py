import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
class TestProgress(test.TestCase):

    def _make_engine(self, flow, flow_detail=None, backend=None):
        e = taskflow.engines.load(flow, flow_detail=flow_detail, backend=backend)
        e.compile()
        e.prepare()
        return e

    def tearDown(self):
        super(TestProgress, self).tearDown()
        with contextlib.closing(impl_memory.MemoryBackend({})) as be:
            with contextlib.closing(be.get_connection()) as conn:
                conn.clear_all()

    def test_sanity_progress(self):
        fired_events = []

        def notify_me(event_type, details):
            fired_events.append(details.pop('progress'))
        ev_count = 5
        t = ProgressTask('test', ev_count)
        t.notifier.register(task.EVENT_UPDATE_PROGRESS, notify_me)
        flo = lf.Flow('test')
        flo.add(t)
        e = self._make_engine(flo)
        e.run()
        self.assertEqual(ev_count + 1, len(fired_events))
        self.assertEqual(1.0, fired_events[-1])
        self.assertEqual(0.0, fired_events[0])

    def test_no_segments_progress(self):
        fired_events = []

        def notify_me(event_type, details):
            fired_events.append(details.pop('progress'))
        t = ProgressTask('test', 0)
        t.notifier.register(task.EVENT_UPDATE_PROGRESS, notify_me)
        flo = lf.Flow('test')
        flo.add(t)
        e = self._make_engine(flo)
        e.run()
        self.assertEqual(2, len(fired_events))
        self.assertEqual(1.0, fired_events[-1])
        self.assertEqual(0.0, fired_events[0])

    def test_storage_progress(self):
        with contextlib.closing(impl_memory.MemoryBackend({})) as be:
            flo = lf.Flow('test')
            flo.add(ProgressTask('test', 3))
            b, fd = p_utils.temporary_flow_detail(be)
            e = self._make_engine(flo, flow_detail=fd, backend=be)
            e.run()
            end_progress = e.storage.get_task_progress('test')
            self.assertEqual(1.0, end_progress)
            task_uuid = e.storage.get_atom_uuid('test')
            td = fd.find(task_uuid)
            self.assertEqual(1.0, td.meta['progress'])
            self.assertFalse(td.meta['progress_details'])

    def test_storage_progress_detail(self):
        flo = ProgressTaskWithDetails('test')
        e = self._make_engine(flo)
        e.run()
        end_progress = e.storage.get_task_progress('test')
        self.assertEqual(1.0, end_progress)
        end_details = e.storage.get_task_progress_details('test')
        self.assertEqual(0.5, end_details.get('at_progress'))
        self.assertEqual({'test': 'test data', 'foo': 'bar'}, end_details.get('details'))

    def test_dual_storage_progress(self):
        fired_events = []

        def notify_me(event_type, details):
            fired_events.append(details.pop('progress'))
        with contextlib.closing(impl_memory.MemoryBackend({})) as be:
            t = ProgressTask('test', 5)
            t.notifier.register(task.EVENT_UPDATE_PROGRESS, notify_me)
            flo = lf.Flow('test')
            flo.add(t)
            b, fd = p_utils.temporary_flow_detail(be)
            e = self._make_engine(flo, flow_detail=fd, backend=be)
            e.run()
            end_progress = e.storage.get_task_progress('test')
            self.assertEqual(1.0, end_progress)
            task_uuid = e.storage.get_atom_uuid('test')
            td = fd.find(task_uuid)
            self.assertEqual(1.0, td.meta['progress'])
            self.assertFalse(td.meta['progress_details'])
            self.assertEqual(6, len(fired_events))