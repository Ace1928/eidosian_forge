from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobConfigurationTableCopy(_messages.Message):
    """A JobConfigurationTableCopy object.

  Fields:
    createDisposition: [Optional] Specifies whether the job is allowed to
      create new tables. The following values are supported: CREATE_IF_NEEDED:
      If the table does not exist, BigQuery creates the table. CREATE_NEVER:
      The table must already exist. If it does not, a 'notFound' error is
      returned in the job result. The default value is CREATE_IF_NEEDED.
      Creation, truncation and append actions occur as one atomic update upon
      job completion.
    destinationTable: [Required] The destination table
    sourceTable: [Pick one] Source table to copy.
    sourceTables: [Pick one] Source tables to copy.
    writeDisposition: [Optional] Specifies the action that occurs if the
      destination table already exists. The following values are supported:
      WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
      table data. WRITE_APPEND: If the table already exists, BigQuery appends
      the data to the table. WRITE_EMPTY: If the table already exists and
      contains data, a 'duplicate' error is returned in the job result. The
      default value is WRITE_EMPTY. Each action is atomic and only occurs if
      BigQuery is able to complete the job successfully. Creation, truncation
      and append actions occur as one atomic update upon job completion.
  """
    createDisposition = _messages.StringField(1)
    destinationTable = _messages.MessageField('TableReference', 2)
    sourceTable = _messages.MessageField('TableReference', 3)
    sourceTables = _messages.MessageField('TableReference', 4, repeated=True)
    writeDisposition = _messages.StringField(5)