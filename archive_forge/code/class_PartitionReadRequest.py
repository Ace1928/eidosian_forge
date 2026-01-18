from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionReadRequest(_messages.Message):
    """The request for PartitionRead

  Fields:
    columns: The columns of table to be returned for each row matching this
      request.
    index: If non-empty, the name of an index on table. This index is used
      instead of the table primary key when interpreting key_set and sorting
      result rows. See key_set for further information.
    keySet: Required. `key_set` identifies the rows to be yielded. `key_set`
      names the primary keys of the rows in table to be yielded, unless index
      is present. If index is present, then key_set instead names index keys
      in index. It is not an error for the `key_set` to name rows that do not
      exist in the database. Read yields nothing for nonexistent rows.
    partitionOptions: Additional options that affect how many partitions are
      created.
    table: Required. The name of the table in the database to be read.
    transaction: Read only snapshot transactions are supported, read/write and
      single use transactions are not.
  """
    columns = _messages.StringField(1, repeated=True)
    index = _messages.StringField(2)
    keySet = _messages.MessageField('KeySet', 3)
    partitionOptions = _messages.MessageField('PartitionOptions', 4)
    table = _messages.StringField(5)
    transaction = _messages.MessageField('TransactionSelector', 6)