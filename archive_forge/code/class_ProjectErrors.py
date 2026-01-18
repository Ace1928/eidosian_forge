from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectErrors(_messages.Message):
    """Provides a summary of the project level error stats.

  Fields:
    internalErrorCount: Optional. Projects that were not validated for
      internal errors and will be automatically retried.
    outsideOrgErrorCount: Optional. Count of projects which are not in the
      same organization.
    outsideOrgProjectNumbers: Optional. Subset of project numbers which are
      not in the same organization.
    validatedCount: Optional. Count of successfully validated projects.
  """
    internalErrorCount = _messages.IntegerField(1)
    outsideOrgErrorCount = _messages.IntegerField(2)
    outsideOrgProjectNumbers = _messages.IntegerField(3, repeated=True)
    validatedCount = _messages.IntegerField(4)