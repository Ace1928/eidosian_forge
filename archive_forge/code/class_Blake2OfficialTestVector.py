import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2OfficialTestVector(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        expected = 'in'
        test_vectors = []
        with open(test_vector_file, 'rt') as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):
                if line.strip() == '' or line.startswith('#'):
                    continue
                res = re.match('%s:\t([0-9A-Fa-f]*)' % expected, line)
                if not res:
                    raise ValueError('Incorrect test vector format (line %d)' % line_number)
                if res.group(1):
                    bin_value = unhexlify(tobytes(res.group(1)))
                else:
                    bin_value = b''
                if expected == 'in':
                    input_data = bin_value
                    expected = 'key'
                elif expected == 'key':
                    key = bin_value
                    expected = 'hash'
                else:
                    result = bin_value
                    expected = 'in'
                    test_vectors.append((input_data, key, result))
        return test_vectors

    def setUp(self):
        dir_comps = ('Hash', self.name)
        file_name = self.name.lower() + '-test.txt'
        self.description = '%s tests' % self.name
        try:
            import pycryptodome_test_vectors
        except ImportError:
            warnings.warn('Warning: skipping extended tests for %s' % self.name, UserWarning)
            self.test_vectors = []
            return
        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        self.test_vectors = self._load_tests(full_file_name)

    def runTest(self):
        for input_data, key, result in self.test_vectors:
            mac = self.BLAKE2.new(key=key, digest_bytes=self.max_bytes)
            mac.update(input_data)
            self.assertEqual(mac.digest(), result)