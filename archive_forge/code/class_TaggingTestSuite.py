import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TaggingTestSuite(TagSetTestCaseBase):

    def testImplicitTag(self):
        t = self.ts1.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 14))
        assert t == tag.TagSet(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 14)), 'implicit tagging went wrong'

    def testExplicitTag(self):
        t = self.ts1.tagExplicitly(tag.Tag(tag.tagClassPrivate, tag.tagFormatSimple, 32))
        assert t == tag.TagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassPrivate, tag.tagFormatConstructed, 32)), 'explicit tagging went wrong'