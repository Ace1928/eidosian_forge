from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupKindDependency(_messages.Message):
    """Defines a dependency between two group kinds.

  Fields:
    requiring: Required. The requiring group kind requires that the other
      group kind be restored first.
    satisfying: Required. The satisfying group kind must be restored first in
      order to satisfy the dependency.
  """
    requiring = _messages.MessageField('GroupKind', 1)
    satisfying = _messages.MessageField('GroupKind', 2)