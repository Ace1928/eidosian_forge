from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
def add_combo(interrepo_cls, from_format, to_format, extra_setup=None, label=None):
    if label is None:
        label = interrepo_cls.__name__
    result.append((label, from_format, to_format, extra_setup))