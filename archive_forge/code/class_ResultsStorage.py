from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultsStorage(_messages.Message):
    """The storage for test results.

  Fields:
    resultsStoragePath: The root directory for test results.
    xunitXmlFile: The path to the Xunit XML file.
  """
    resultsStoragePath = _messages.MessageField('FileReference', 1)
    xunitXmlFile = _messages.MessageField('FileReference', 2)