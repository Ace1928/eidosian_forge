from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidTest(_messages.Message):
    """An Android mobile test specification.

  Fields:
    androidAppInfo: Information about the application under test.
    androidInstrumentationTest: An Android instrumentation test.
    androidRoboTest: An Android robo test.
    androidTestLoop: An Android test loop.
    testTimeout: Max time a test is allowed to run before it is automatically
      cancelled.
  """
    androidAppInfo = _messages.MessageField('AndroidAppInfo', 1)
    androidInstrumentationTest = _messages.MessageField('AndroidInstrumentationTest', 2)
    androidRoboTest = _messages.MessageField('AndroidRoboTest', 3)
    androidTestLoop = _messages.MessageField('AndroidTestLoop', 4)
    testTimeout = _messages.MessageField('Duration', 5)