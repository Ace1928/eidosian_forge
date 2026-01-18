from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiStep(_messages.Message):
    """Details when multiple steps are run with the same configuration as a
  group.

  Fields:
    multistepNumber: Unique int given to each step. Ranges from 0(inclusive)
      to total number of steps(exclusive). The primary step is 0.
    primaryStep: Present if it is a primary (original) step.
    primaryStepId: Step Id of the primary (original) step, which might be this
      step.
  """
    multistepNumber = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    primaryStep = _messages.MessageField('PrimaryStep', 2)
    primaryStepId = _messages.StringField(3)