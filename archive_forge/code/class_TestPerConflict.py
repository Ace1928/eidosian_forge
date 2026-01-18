import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
class TestPerConflict(tests.TestCase):
    scenarios = scenarios.multiply_scenarios(vary_by_conflicts())

    def test_stringification(self):
        text = str(self.conflict)
        self.assertContainsString(text, self.conflict.path)
        self.assertContainsString(text.lower(), 'conflict')
        self.assertContainsString(repr(self.conflict), self.conflict.__class__.__name__)

    def test_stanza_roundtrip(self):
        p = self.conflict
        o = bzr_conflicts.Conflict.factory(**p.as_stanza().as_dict())
        self.assertEqual(o, p)
        self.assertIsInstance(o.path, str)
        if o.file_id is not None:
            self.assertIsInstance(o.file_id, bytes)
        conflict_path = getattr(o, 'conflict_path', None)
        if conflict_path is not None:
            self.assertIsInstance(conflict_path, str)
        conflict_file_id = getattr(o, 'conflict_file_id', None)
        if conflict_file_id is not None:
            self.assertIsInstance(conflict_file_id, bytes)

    def test_stanzification(self):
        stanza = self.conflict.as_stanza()
        if 'file_id' in stanza:
            self.assertStartsWith(stanza['file_id'], 'îd')
        self.assertStartsWith(stanza['path'], 'påth')
        if 'conflict_path' in stanza:
            self.assertStartsWith(stanza['conflict_path'], 'påth')
        if 'conflict_file_id' in stanza:
            self.assertStartsWith(stanza['conflict_file_id'], 'îd')