import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
def _check_threads(self):
    active = threading.activeCount()
    if active > 1:
        self.knownFailure('%d active threads, bug #103133 needs to be fixed.' % active)