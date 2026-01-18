import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def _import_object(source):
    """helper to import object from module; accept format `path.to.object`"""
    modname, modattr = source.rsplit('.', 1)
    mod = __import__(modname, fromlist=[modattr], level=0)
    return getattr(mod, modattr)