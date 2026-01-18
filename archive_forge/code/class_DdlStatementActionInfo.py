from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DdlStatementActionInfo(_messages.Message):
    """Action information extracted from a DDL statement. This proto is used to
  display the brief info of the DDL statement for the operation
  UpdateDatabaseDdl.

  Fields:
    action: The action for the DDL statement, e.g. CREATE, ALTER, DROP, GRANT,
      etc. This field is a non-empty string.
    entityNames: The entity name(s) being operated on the DDL statement. E.g.
      1. For statement "CREATE TABLE t1(...)", `entity_names` = ["t1"]. 2. For
      statement "GRANT ROLE r1, r2 ...", `entity_names` = ["r1", "r2"]. 3. For
      statement "ANALYZE", `entity_names` = [].
    entityType: The entity type for the DDL statement, e.g. TABLE, INDEX,
      VIEW, etc. This field can be empty string for some DDL statement, e.g.
      for statement "ANALYZE", `entity_type` = "".
  """
    action = _messages.StringField(1)
    entityNames = _messages.StringField(2, repeated=True)
    entityType = _messages.StringField(3)