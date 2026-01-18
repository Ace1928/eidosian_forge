from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobConfiguration(_messages.Message):
    """A JobConfiguration object.

  Fields:
    copy: [Pick one] Copies a table.
    dryRun: [Optional] If set, don't actually run this job. A valid query will
      return a mostly empty response with some processing statistics, while an
      invalid query will return the same error it would if it wasn't a dry
      run. Behavior of non-query jobs is undefined.
    extract: [Pick one] Configures an extract job.
    load: [Pick one] Configures a load job.
    query: [Pick one] Configures a query job.
  """
    copy = _messages.MessageField('JobConfigurationTableCopy', 1)
    dryRun = _messages.BooleanField(2)
    extract = _messages.MessageField('JobConfigurationExtract', 3)
    load = _messages.MessageField('JobConfigurationLoad', 4)
    query = _messages.MessageField('JobConfigurationQuery', 5)