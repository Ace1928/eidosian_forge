import hashlib
import os
import random
from operator import attrgetter
import gyp.common
def MakeGuid(name, seed='msvs_new'):
    """Returns a GUID for the specified target name.

  Args:
    name: Target name.
    seed: Seed for MD5 hash.
  Returns:
    A GUID-line string calculated from the name and seed.

  This generates something which looks like a GUID, but depends only on the
  name and seed.  This means the same name/seed will always generate the same
  GUID, so that projects and solutions which refer to each other can explicitly
  determine the GUID to refer to explicitly.  It also means that the GUID will
  not change when the project for a target is rebuilt.
  """
    d = hashlib.md5((str(seed) + str(name)).encode('utf-8')).hexdigest().upper()
    guid = '{' + d[:8] + '-' + d[8:12] + '-' + d[12:16] + '-' + d[16:20] + '-' + d[20:32] + '}'
    return guid