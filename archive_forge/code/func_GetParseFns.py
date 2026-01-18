from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def GetParseFns(fn):
    metadata = GetMetadata(fn)
    default = {'default': None, 'positional': [], 'named': {}}
    return metadata.get(FIRE_PARSE_FNS, default)