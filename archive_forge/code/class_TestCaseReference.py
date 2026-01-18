from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestCaseReference(_messages.Message):
    """A reference to a test case. Test case references are canonically ordered
  lexicographically by these three factors: * First, by test_suite_name. *
  Second, by class_name. * Third, by name.

  Fields:
    className: The name of the class.
    name: The name of the test case. Required.
    testSuiteName: The name of the test suite to which this test case belongs.
  """
    className = _messages.StringField(1)
    name = _messages.StringField(2)
    testSuiteName = _messages.StringField(3)