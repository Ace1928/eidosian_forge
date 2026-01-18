import functools
import inspect
import wrapt
from debtcollector import _utils
def _cls_decorator(cls):
    _check_it(cls)
    out_message = _utils.generate_message("Using class '%s' (either directly or via inheritance) is deprecated" % cls_name, postfix=None, message=message, version=version, removal_version=removal_version)
    cls.__init__ = _wrap_it(cls.__init__, out_message)
    return cls