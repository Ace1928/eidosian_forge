import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagSetTestCaseBase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.ts1 = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))
        self.ts2 = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))