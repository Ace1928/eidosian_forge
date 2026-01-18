from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InitialStateConfig(_messages.Message):
    """Initial State for shielded instance, these are public keys which are
  safe to store in public

  Fields:
    dbs: The Key Database (db).
    dbxs: The forbidden key database (dbx).
    keks: The Key Exchange Key (KEK).
    pk: The Platform Key (PK).
  """
    dbs = _messages.MessageField('FileContentBuffer', 1, repeated=True)
    dbxs = _messages.MessageField('FileContentBuffer', 2, repeated=True)
    keks = _messages.MessageField('FileContentBuffer', 3, repeated=True)
    pk = _messages.MessageField('FileContentBuffer', 4)