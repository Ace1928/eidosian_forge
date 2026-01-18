import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def _add_except(exc, info):
    if not hasattr(exc, 'args') or exc.args is None:
        return
    args = list(exc.args)
    if args:
        args[0] += ' ' + info
    else:
        args = [info]
    exc.args = tuple(args)
    return