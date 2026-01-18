import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
class TestDnsTests(ExampleTestBase, TestCase):
    """
    Test the testdns.py example script.
    """
    exampleRelativePath = 'names/examples/testdns.py'