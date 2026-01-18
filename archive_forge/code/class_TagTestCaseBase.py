import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagTestCaseBase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.t1 = tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 3)
        self.t2 = tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 3)