import copy
import re
from collections import namedtuple
def _mergeOpts(options, childFieldName):
    if options is None:
        options = {}
    if isinstance(options, tuple):
        options = dict(options)
    options = _normalizeOpts(options)
    finalOpts = copy.copy(options)
    if isinstance(options, dict):
        local = finalOpts.get(childFieldName, None)
        if local:
            del finalOpts[childFieldName]
            for key in local:
                finalOpts[key] = local[key]
        finalOpts = namedtuple('CustomOptions', finalOpts.keys())(*finalOpts.values())
    if isinstance(options, Options):
        local = getattr(finalOpts, childFieldName, None)
        if local:
            delattr(finalOpts, childFieldName)
            for key in local:
                setattr(finalOpts, key, local[key])
    return finalOpts