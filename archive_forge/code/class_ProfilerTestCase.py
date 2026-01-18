import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class ProfilerTestCase(test.TestCase):

    def test_profiler_get_shorten_id(self):
        uuid_id = '4e3e0ec6-2938-40b1-8504-09eb1d4b0dee'
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        result = prof.get_shorten_id(uuid_id)
        expected = '850409eb1d4b0dee'
        self.assertEqual(expected, result)

    def test_profiler_get_shorten_id_int(self):
        short_id_int = 42
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        result = prof.get_shorten_id(short_id_int)
        expected = '2a'
        self.assertEqual(expected, result)

    def test_profiler_get_base_id(self):
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        self.assertEqual(prof.get_base_id(), '1')

    @mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
    def test_profiler_get_parent_id(self, mock_generate_uuid):
        mock_generate_uuid.return_value = '42'
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        prof.start('test')
        self.assertEqual(prof.get_parent_id(), '2')

    @mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
    def test_profiler_get_base_id_unset_case(self, mock_generate_uuid):
        mock_generate_uuid.return_value = '42'
        prof = profiler._Profiler('secret')
        self.assertEqual(prof.get_base_id(), '42')
        self.assertEqual(prof.get_parent_id(), '42')

    @mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
    def test_profiler_get_id(self, mock_generate_uuid):
        mock_generate_uuid.return_value = '43'
        prof = profiler._Profiler('secret')
        prof.start('test')
        self.assertEqual(prof.get_id(), '43')

    @mock.patch('osprofiler.profiler.datetime')
    @mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
    @mock.patch('osprofiler.profiler.notifier.notify')
    def test_profiler_start(self, mock_notify, mock_generate_uuid, mock_datetime):
        mock_generate_uuid.return_value = '44'
        now = datetime.datetime.utcnow()
        mock_datetime.datetime.utcnow.return_value = now
        info = {'some': 'info'}
        payload = {'name': 'test-start', 'base_id': '1', 'parent_id': '2', 'trace_id': '44', 'info': info, 'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%f')}
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        prof.start('test', info=info)
        mock_notify.assert_called_once_with(payload)

    @mock.patch('osprofiler.profiler.datetime')
    @mock.patch('osprofiler.profiler.notifier.notify')
    def test_profiler_stop(self, mock_notify, mock_datetime):
        now = datetime.datetime.utcnow()
        mock_datetime.datetime.utcnow.return_value = now
        prof = profiler._Profiler('secret', base_id='1', parent_id='2')
        prof._trace_stack.append('44')
        prof._name.append('abc')
        info = {'some': 'info'}
        prof.stop(info=info)
        payload = {'name': 'abc-stop', 'base_id': '1', 'parent_id': '2', 'trace_id': '44', 'info': info, 'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%f')}
        mock_notify.assert_called_once_with(payload)
        self.assertEqual(len(prof._name), 0)
        self.assertEqual(prof._trace_stack, collections.deque(['1', '2']))

    def test_profiler_hmac(self):
        hmac = 'secret'
        prof = profiler._Profiler(hmac, base_id='1', parent_id='2')
        self.assertEqual(hmac, prof.hmac_key)