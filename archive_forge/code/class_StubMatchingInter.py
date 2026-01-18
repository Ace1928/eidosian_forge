from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
class StubMatchingInter:
    """An inter for tests.

    This is not a subclass of InterBranch so that missing methods are caught
    and added rather than actually trying to do something.
    """
    _uses: List[Any] = []

    def __init__(self, source, target):
        self.source = source
        self.target = target

    @classmethod
    def is_compatible(klass, source, target):
        return StubWithFormat._format in (source._format, target._format)

    def copy_content_into(self, *args, **kwargs):
        self.__class__._uses.append((self, 'copy_content_into', args, kwargs))