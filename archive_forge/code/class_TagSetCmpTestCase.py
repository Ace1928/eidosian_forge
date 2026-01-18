import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagSetCmpTestCase(TagSetTestCaseBase):

    def testCmp(self):
        assert self.ts1 == self.ts2, 'tag set comparation fails'

    def testHash(self):
        assert hash(self.ts1) == hash(self.ts2), 'tag set hash comp. fails'

    def testLen(self):
        assert len(self.ts1) == len(self.ts2), 'tag length comparation fails'