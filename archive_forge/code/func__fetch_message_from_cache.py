import functools
import inspect
import wrapt
from debtcollector import _utils
def _fetch_message_from_cache(self, kind):
    try:
        out_message = self._message_cache[kind]
    except KeyError:
        prefix_tpl = self._PROPERTY_GONE_TPLS[kind]
        prefix = prefix_tpl % _fetch_first_result(self.fget, self.fset, self.fdel, _get_qualified_name, value_not_found='???')
        out_message = _utils.generate_message(prefix, message=self.message, version=self.version, removal_version=self.removal_version)
        self._message_cache[kind] = out_message
    return out_message