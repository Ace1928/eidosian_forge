import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def _test_merge_from_strings(self, base, a, b, expected):
    w = self.get_file()
    w.add_lines(b'text0', [], base.splitlines(True))
    w.add_lines(b'text1', [b'text0'], a.splitlines(True))
    w.add_lines(b'text2', [b'text0'], b.splitlines(True))
    self.log('merge plan:')
    p = list(w.plan_merge(b'text1', b'text2'))
    for state, line in p:
        if line:
            self.log('%12s | %s' % (state, line[:-1]))
    self.log('merge result:')
    result_text = b''.join(w.weave_merge(p))
    self.log(result_text)
    self.assertEqualDiff(result_text, expected)