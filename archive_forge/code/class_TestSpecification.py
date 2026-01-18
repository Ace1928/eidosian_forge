from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestSpecification(_messages.Message):
    """A description of how to run the test.

  Fields:
    androidInstrumentationTest: An Android instrumentation test.
    androidRoboTest: An Android robo test.
    androidTestLoop: An Android Application with a Test Loop.
    disablePerformanceMetrics: Disables performance metrics recording. May
      reduce test latency.
    disableVideoRecording: Disables video recording. May reduce test latency.
    iosRoboTest: An iOS Robo test.
    iosTestLoop: An iOS application with a test loop.
    iosTestSetup: Test setup requirements for iOS.
    iosXcTest: An iOS XCTest, via an .xctestrun file.
    testSetup: Test setup requirements for Android e.g. files to install,
      bootstrap scripts.
    testTimeout: Max time a test execution is allowed to run before it is
      automatically cancelled. The default value is 5 min.
  """
    androidInstrumentationTest = _messages.MessageField('AndroidInstrumentationTest', 1)
    androidRoboTest = _messages.MessageField('AndroidRoboTest', 2)
    androidTestLoop = _messages.MessageField('AndroidTestLoop', 3)
    disablePerformanceMetrics = _messages.BooleanField(4)
    disableVideoRecording = _messages.BooleanField(5)
    iosRoboTest = _messages.MessageField('IosRoboTest', 6)
    iosTestLoop = _messages.MessageField('IosTestLoop', 7)
    iosTestSetup = _messages.MessageField('IosTestSetup', 8)
    iosXcTest = _messages.MessageField('IosXcTest', 9)
    testSetup = _messages.MessageField('TestSetup', 10)
    testTimeout = _messages.StringField(11)