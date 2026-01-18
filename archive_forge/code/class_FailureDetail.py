from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailureDetail(_messages.Message):
    """Details for an outcome with a FAILURE outcome summary.

  Fields:
    crashed: If the failure was severe because the system (app) under test
      crashed.
    deviceOutOfMemory: If the device ran out of memory during a test, causing
      the test to crash.
    failedRoboscript: If the Roboscript failed to complete successfully, e.g.,
      because a Roboscript action or assertion failed or a Roboscript action
      could not be matched during the entire crawl.
    notInstalled: If an app is not installed and thus no test can be run with
      the app. This might be caused by trying to run a test on an unsupported
      platform.
    otherNativeCrash: If a native process (including any other than the app)
      crashed.
    timedOut: If the test overran some time limit, and that is why it failed.
    unableToCrawl: If the robo was unable to crawl the app; perhaps because
      the app did not start.
  """
    crashed = _messages.BooleanField(1)
    deviceOutOfMemory = _messages.BooleanField(2)
    failedRoboscript = _messages.BooleanField(3)
    notInstalled = _messages.BooleanField(4)
    otherNativeCrash = _messages.BooleanField(5)
    timedOut = _messages.BooleanField(6)
    unableToCrawl = _messages.BooleanField(7)