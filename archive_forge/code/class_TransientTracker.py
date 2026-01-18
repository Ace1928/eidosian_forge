from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TransientTracker:
    """An transient tracker used for testing."""

    @classmethod
    def get(klass, abbreviation, branch):
        klass.log.append(('get', abbreviation, branch))
        if abbreviation != 'transient':
            return None
        return klass()

    def get_bug_url(self, bug_id):
        self.log.append(('get_bug_url', bug_id))
        return 'http://bugs.example.com/%s' % bug_id