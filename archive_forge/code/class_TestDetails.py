from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestDetails(_messages.Message):
    """Additional details about the progress of the running test.

  Fields:
    errorMessage: Output only. If the TestState is ERROR, then this string
      will contain human-readable details about the error.
    progressMessages: Output only. Human-readable, detailed descriptions of
      the test's progress. For example: "Provisioning a device", "Starting
      Test". During the course of execution new data may be appended to the
      end of progress_messages.
  """
    errorMessage = _messages.StringField(1)
    progressMessages = _messages.StringField(2, repeated=True)