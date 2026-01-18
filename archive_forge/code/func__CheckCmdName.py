from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def _CheckCmdName(name_or_alias):
    """Only allow strings for command names and aliases (reject unicode as well).

  Args:
    name_or_alias: properly formatted string name or alias.

  Raises:
    AppCommandsError: is command is already registered OR cmd is not a subclass
                      of Cmd
    AppCommandsError: if name is already registered OR name is not a string OR
                      name is too short OR name does not start with a letter OR
                      name contains any non alphanumeric characters besides
                      '_', '-', or ':'.
  """
    if name_or_alias in GetCommandAliasList():
        raise AppCommandsError("Command or Alias '%s' already defined" % name_or_alias)
    if not isinstance(name_or_alias, str) or len(name_or_alias) <= 1:
        raise AppCommandsError("Command '%s' not a string or too short" % str(name_or_alias))
    if not name_or_alias[0].isalpha():
        raise AppCommandsError("Command '%s' does not start with a letter" % name_or_alias)
    if [c for c in name_or_alias if not (c.isalnum() or c in ('_', '-', ':'))]:
        raise AppCommandsError("Command '%s' contains non alphanumeric characters" % name_or_alias)