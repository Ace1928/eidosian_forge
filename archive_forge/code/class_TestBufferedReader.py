import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import backend
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
class TestBufferedReader(base.StoreBaseTest):
    _CONF = cfg.CONF

    def setUp(self):
        super(TestBufferedReader, self).setUp()
        self.config(swift_upload_buffer_dir=self.test_dir)
        s = b'1234567890'
        self.infile = io.BytesIO(s)
        self.infile.seek(0)
        self.checksum = md5(usedforsecurity=False)
        self.hash_algo = HASH_ALGO
        self.os_hash_value = hashlib.sha256()
        self.verifier = mock.MagicMock(name='mock_verifier')
        total = 7
        self.reader = buffered.BufferedReader(self.infile, self.checksum, self.os_hash_value, total, self.verifier)
        self.addCleanup(self.conf.reset)

    def tearDown(self):
        super(TestBufferedReader, self).tearDown()
        self.reader.__exit__(None, None, None)

    def test_buffer(self):
        self.reader.read(4)
        self.assertTrue(self.reader._buffered)
        self.assertEqual(4, self.reader.tell())
        buf = self.reader._tmpfile
        buf.seek(0)
        self.assertEqual(b'1234567', buf.read())

    def test_read(self):
        buf = self.reader.read(4)
        self.assertEqual(b'1234', buf)
        buf = self.reader.read(4)
        self.assertEqual(b'567', buf)
        self.assertEqual(7, self.reader.tell())

    def test_read_limited(self):
        self.assertEqual(b'1234567', self.reader.read(100))

    def test_reset(self):
        self.assertEqual(0, self.reader.tell())
        self.reader.read(4)
        self.assertEqual(4, self.reader.tell())
        self.reader.seek(0)
        self.assertEqual(0, self.reader.tell())
        self.assertEqual(b'1234', self.reader.read(4))

    def test_partial_reset(self):
        self.reader.read(4)
        self.reader.seek(2)
        self.assertEqual(b'34567', self.reader.read(10))

    def test_checksums(self):
        expected_csum = md5(usedforsecurity=False)
        expected_csum.update(b'1234567')
        expected_multihash = hashlib.sha256()
        expected_multihash.update(b'1234567')
        self.reader.read(7)
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())

    def test_checksum_updated_only_once_w_full_segment_read(self):
        expected_csum = md5(usedforsecurity=False)
        expected_csum.update(b'1234567')
        expected_multihash = hashlib.sha256()
        expected_multihash.update(b'1234567')
        self.reader.read(7)
        self.reader.seek(4)
        self.reader.read(1)
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())

    def test_checksum_updates_during_partial_segment_reads(self):
        expected_csum = md5(usedforsecurity=False)
        expected_multihash = hashlib.sha256()
        self.reader.read(4)
        expected_csum.update(b'1234')
        expected_multihash.update(b'1234')
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())
        self.reader.seek(0)
        self.reader.read(2)
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())
        self.reader.read(4)
        expected_csum.update(b'56')
        expected_multihash.update(b'56')
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())

    def test_checksum_rolling_calls(self):
        expected_csum = md5(usedforsecurity=False)
        expected_multihash = hashlib.sha256()
        self.reader.read(7)
        expected_csum.update(b'1234567')
        expected_multihash.update(b'1234567')
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())
        reader1 = buffered.BufferedReader(self.infile, self.checksum, self.os_hash_value, 3, self.reader.verifier)
        reader1.read(3)
        expected_csum.update(b'890')
        expected_multihash.update(b'890')
        self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
        self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())

    def test_verifier(self):
        self.reader.read(7)
        self.verifier.update.assert_called_once_with(b'1234567')

    def test_verifier_updated_only_once_w_full_segment_read(self):
        self.reader.read(7)
        self.reader.seek(4)
        self.reader.read(5)
        self.verifier.update.assert_called_once_with(b'1234567')

    def test_verifier_updates_during_partial_segment_reads(self):
        self.reader.read(4)
        self.verifier.update.assert_called_once_with(b'1234')
        self.reader.seek(0)
        self.reader.read(2)
        self.verifier.update.assert_called_once_with(b'1234')
        self.reader.read(4)
        self.verifier.update.assert_called_with(b'56')
        self.assertEqual(2, self.verifier.update.call_count)

    def test_verifier_rolling_calls(self):
        self.reader.read(7)
        self.verifier.update.assert_called_once_with(b'1234567')
        self.assertEqual(1, self.verifier.update.call_count)
        reader1 = buffered.BufferedReader(self.infile, self.checksum, self.os_hash_value, 3, self.reader.verifier)
        reader1.read(3)
        self.verifier.update.assert_called_with(b'890')
        self.assertEqual(2, self.verifier.update.call_count)

    def test_light_buffer(self):
        s = b'12'
        infile = io.BytesIO(s)
        infile.seek(0)
        total = 7
        checksum = md5(usedforsecurity=False)
        os_hash_value = hashlib.sha256()
        self.reader = buffered.BufferedReader(infile, checksum, os_hash_value, total)
        self.reader.read(0)
        self.assertEqual(b'12', self.reader.read(7))
        self.assertEqual(2, self.reader.tell())

    def test_context_exit(self):
        with self.reader:
            pass
        if getattr(self.reader._tmpfile, 'closed'):
            self.assertTrue(self.reader._tmpfile.closed)

    def test_read_all_data(self):
        """
        Replicate what goes on in the Swift driver with the
        repeated creation of the BufferedReader object
        """
        CHUNKSIZE = 100
        data = b'*' * units.Ki
        expected_checksum = md5(data, usedforsecurity=False).hexdigest()
        expected_multihash = hashlib.sha256(data).hexdigest()
        data_file = tempfile.NamedTemporaryFile()
        data_file.write(data)
        data_file.flush()
        infile = open(data_file.name, 'rb')
        bytes_read = 0
        checksum = md5(usedforsecurity=False)
        os_hash_value = hashlib.sha256()
        while True:
            cr = buffered.BufferedReader(infile, checksum, os_hash_value, CHUNKSIZE)
            chunk = cr.read(CHUNKSIZE)
            if len(chunk) == 0:
                self.assertEqual(True, cr.is_zero_size)
                break
            else:
                self.assertEqual(False, cr.is_zero_size)
            bytes_read += len(chunk)
        self.assertEqual(units.Ki, bytes_read)
        self.assertEqual(expected_checksum, cr.checksum.hexdigest())
        self.assertEqual(expected_multihash, cr.os_hash_value.hexdigest())
        data_file.close()
        infile.close()