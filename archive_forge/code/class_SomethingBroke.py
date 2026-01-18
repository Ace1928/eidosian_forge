import unittest
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures import TestWithFixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
class SomethingBroke(Exception):
    pass