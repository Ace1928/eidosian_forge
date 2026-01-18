import unittest
from mock import Mock, call
from .watch import Watch
class WatchTests(unittest.TestCase):

    def setUp(self):
        self.callcount = 0

    def test_watch_with_decode(self):
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": {"metadata": {"name": "test1","resourceVersion": "1"}, "spec": {}, "status": {}}}\n', '{"type": "ADDED", "object": {"metadata": {"name": "test2","resourceVersion": "2"}, "spec": {}, "sta', 'tus": {}}}\n{"type": "ADDED", "object": {"metadata": {"name": "test3","resourceVersion": "3"}, "spec": {}, "status": {}}}\n', 'should_not_happened\n'])
        fake_api = Mock()
        fake_api.get_namespaces = Mock(return_value=fake_resp)
        fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
        w = Watch()
        count = 1
        for e in w.stream(fake_api.get_namespaces):
            self.assertEqual('ADDED', e['type'])
            self.assertEqual('test%d' % count, e['object'].metadata.name)
            self.assertEqual('%d' % count, e['object'].metadata.resource_version)
            self.assertEqual('%d' % count, w.resource_version)
            count += 1
            if count == 4:
                w.stop()
        fake_api.get_namespaces.assert_called_once_with(_preload_content=False, watch=True)
        fake_resp.read_chunked.assert_called_once_with(decode_content=False)
        fake_resp.close.assert_called_once()
        fake_resp.release_conn.assert_called_once()

    def test_watch_for_follow(self):
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        fake_resp.read_chunked = Mock(return_value=['log_line_1\n', 'log_line_2\n'])
        fake_api = Mock()
        fake_api.read_namespaced_pod_log = Mock(return_value=fake_resp)
        fake_api.read_namespaced_pod_log.__doc__ = ':param bool follow:\n:return: str'
        w = Watch()
        count = 1
        for e in w.stream(fake_api.read_namespaced_pod_log):
            self.assertEqual('log_line_1', e)
            count += 1
            if count == 2:
                w.stop()
        fake_api.read_namespaced_pod_log.assert_called_once_with(_preload_content=False, follow=True)
        fake_resp.read_chunked.assert_called_once_with(decode_content=False)
        fake_resp.close.assert_called_once()
        fake_resp.release_conn.assert_called_once()

    def test_watch_resource_version_set(self):
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        values = ['{"type": "ADDED", "object": {"metadata": {"name": "test1","resourceVersion": "1"}, "spec": {}, "status": {}}}\n', '{"type": "ADDED", "object": {"metadata": {"name": "test2","resourceVersion": "2"}, "spec": {}, "sta', 'tus": {}}}\n{"type": "ADDED", "object": {"metadata": {"name": "test3","resourceVersion": "3"}, "spec": {}, "status": {}}}\n']

        def get_values(*args, **kwargs):
            self.callcount += 1
            if self.callcount == 1:
                return []
            else:
                return values
        fake_resp.read_chunked = Mock(side_effect=get_values)
        fake_api = Mock()
        fake_api.get_namespaces = Mock(return_value=fake_resp)
        fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
        w = Watch()
        calls = []
        iterations = 2
        calls.append(call(_preload_content=False, watch=True, resource_version='5'))
        calls.append(call(_preload_content=False, watch=True, resource_version='5'))
        for i in range(iterations):
            calls.append(call(_preload_content=False, watch=True, resource_version='3'))
        for c, e in enumerate(w.stream(fake_api.get_namespaces, resource_version='5')):
            if c == len(values) * iterations:
                w.stop()
        fake_api.get_namespaces.assert_has_calls(calls)
        self.assertEqual(fake_api.get_namespaces.mock_calls, calls)

    def test_watch_stream_twice(self):
        w = Watch(float)
        for step in ['first', 'second']:
            fake_resp = Mock()
            fake_resp.close = Mock()
            fake_resp.release_conn = Mock()
            fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": 1}\n'] * 4)
            fake_api = Mock()
            fake_api.get_namespaces = Mock(return_value=fake_resp)
            fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
            count = 1
            for e in w.stream(fake_api.get_namespaces):
                count += 1
                if count == 3:
                    w.stop()
            self.assertEqual(count, 3)
            fake_api.get_namespaces.assert_called_once_with(_preload_content=False, watch=True)
            fake_resp.read_chunked.assert_called_once_with(decode_content=False)
            fake_resp.close.assert_called_once()
            fake_resp.release_conn.assert_called_once()

    def test_watch_stream_loop(self):
        w = Watch(float)
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": 1}\n'])
        fake_api = Mock()
        fake_api.get_namespaces = Mock(return_value=fake_resp)
        fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
        count = 0
        for e in w.stream(fake_api.get_namespaces, timeout_seconds=1):
            count = count + 1
        self.assertEqual(count, 1)
        for e in w.stream(fake_api.get_namespaces):
            count = count + 1
            if count == 2:
                w.stop()
        self.assertEqual(count, 2)
        self.assertEqual(fake_api.get_namespaces.call_count, 2)
        self.assertEqual(fake_resp.read_chunked.call_count, 2)
        self.assertEqual(fake_resp.close.call_count, 2)
        self.assertEqual(fake_resp.release_conn.call_count, 2)

    def test_unmarshal_with_float_object(self):
        w = Watch()
        event = w.unmarshal_event('{"type": "ADDED", "object": 1}', 'float')
        self.assertEqual('ADDED', event['type'])
        self.assertEqual(1.0, event['object'])
        self.assertTrue(isinstance(event['object'], float))
        self.assertEqual(1, event['raw_object'])

    def test_unmarshal_with_no_return_type(self):
        w = Watch()
        event = w.unmarshal_event('{"type": "ADDED", "object": ["test1"]}', None)
        self.assertEqual('ADDED', event['type'])
        self.assertEqual(['test1'], event['object'])
        self.assertEqual(['test1'], event['raw_object'])

    def test_unmarshal_with_custom_object(self):
        w = Watch()
        event = w.unmarshal_event('{"type": "ADDED", "object": {"apiVersion":"test.com/v1beta1","kind":"foo","metadata":{"name": "bar", "resourceVersion": "1"}}}', 'object')
        self.assertEqual('ADDED', event['type'])
        self.assertTrue(isinstance(event['object'], dict))
        self.assertEqual('1', event['object']['metadata']['resourceVersion'])
        self.assertEqual('1', w.resource_version)

    def test_watch_with_exception(self):
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        fake_resp.read_chunked = Mock(side_effect=KeyError('expected'))
        fake_api = Mock()
        fake_api.get_thing = Mock(return_value=fake_resp)
        w = Watch()
        try:
            for _ in w.stream(fake_api.get_thing):
                self.fail(self, 'Should fail on exception.')
        except KeyError:
            pass
        fake_api.get_thing.assert_called_once_with(_preload_content=False, watch=True)
        fake_resp.read_chunked.assert_called_once_with(decode_content=False)
        fake_resp.close.assert_called_once()
        fake_resp.release_conn.assert_called_once()