import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagSetReprTestCase(TagSetTestCaseBase):

    def testRepr(self):
        assert 'TagSet' in repr(self.ts1)