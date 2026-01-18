from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NavigationInfo(_messages.Message):
    """NavigationInfo describes what steps if any come before or after this
  step, or what steps are parents or children of this step.

  Fields:
    children: Step entries that can be reached by "stepping into" e.g. a
      subworkflow call.
    next: The index of the next step in the current workflow, if any.
    parent: The step entry, if any, that can be reached by "stepping out" of
      the current workflow being executed.
    previous: The index of the previous step in the current workflow, if any.
  """
    children = _messages.IntegerField(1, repeated=True)
    next = _messages.IntegerField(2)
    parent = _messages.IntegerField(3)
    previous = _messages.IntegerField(4)