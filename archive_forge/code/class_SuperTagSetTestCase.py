import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class SuperTagSetTestCase(TagSetTestCaseBase):

    def testSuperTagCheck1(self):
        assert self.ts1.isSuperTagSetOf(tag.TagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))), 'isSuperTagSetOf() fails'

    def testSuperTagCheck2(self):
        assert not self.ts1.isSuperTagSetOf(tag.TagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 13))), 'isSuperTagSetOf() fails'

    def testSuperTagCheck3(self):
        assert self.ts1.isSuperTagSetOf(tag.TagSet((), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))), 'isSuperTagSetOf() fails'