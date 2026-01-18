from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidRoboTest(_messages.Message):
    """A test of an android application that explores the application on a
  virtual or physical Android device, finding culprits and crashes as it goes.

  Fields:
    appInitialActivity: The initial activity that should be used to start the
      app. Optional
    bootstrapPackageId: The java package for the bootstrap. Optional
    bootstrapRunnerClass: The runner class for the bootstrap. Optional
    maxDepth: The max depth of the traversal stack Robo can explore. Optional
    maxSteps: The max number of steps/actions Robo can execute. Default is no
      limit (0). Optional
  """
    appInitialActivity = _messages.StringField(1)
    bootstrapPackageId = _messages.StringField(2)
    bootstrapRunnerClass = _messages.StringField(3)
    maxDepth = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    maxSteps = _messages.IntegerField(5, variant=_messages.Variant.INT32)