import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
class GraftSerializerTests(TestCase):

    def assertSerialize(self, expected, graftpoints):
        self.assertEqual(sorted(expected), sorted(serialize_graftpoints(graftpoints)))

    def test_no_grafts(self):
        self.assertSerialize(b'', {})

    def test_no_parents(self):
        self.assertSerialize(makesha(0), {makesha(0): []})

    def test_parents(self):
        self.assertSerialize(b' '.join([makesha(0), makesha(1), makesha(2)]), {makesha(0): [makesha(1), makesha(2)]})

    def test_multiple_hybrid(self):
        self.assertSerialize(b'\n'.join([makesha(0), b' '.join([makesha(1), makesha(2)]), b' '.join([makesha(3), makesha(4), makesha(5)])]), {makesha(0): [], makesha(1): [makesha(2)], makesha(3): [makesha(4), makesha(5)]})