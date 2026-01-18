import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
class DnsServiceTests(ExampleTestBase, TestCase):
    """
    Test the dns-service.py example script.
    """
    exampleRelativePath = 'names/examples/dns-service.py'