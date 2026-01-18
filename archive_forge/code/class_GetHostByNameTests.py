import os
import sys
from io import StringIO
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SkipTest, TestCase
class GetHostByNameTests(ExampleTestBase, TestCase):
    """
    Test the gethostbyname.py example script.
    """
    exampleRelativePath = 'names/examples/gethostbyname.py'