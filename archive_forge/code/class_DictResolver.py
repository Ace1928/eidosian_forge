from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class DictResolver:
    sort_keys = not IS_PY36_OR_GREATER

    def resolve(self, dct, key):
        if key in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None
        if '(' not in key:
            try:
                return dct[key]
            except:
                return getattr(dct, key)
        expected_id = int(key.split('(')[-1][:-1])
        for key, val in dct.items():
            if id(key) == expected_id:
                return val
        raise UnableToResolveVariableException()

    def key_to_str(self, key, fmt=None):
        if fmt is not None:
            if fmt.get('hex', False):
                safe_repr = SafeRepr()
                safe_repr.convert_to_hex = True
                return safe_repr(key)
        return '%r' % (key,)

    def init_dict(self):
        return {}

    def get_contents_debug_adapter_protocol(self, dct, fmt=None):
        """
        This method is to be used in the case where the variables are all saved by its id (and as
        such don't need to have the `resolve` method called later on, so, keys don't need to
        embed the reference in the key).

        Note that the return should be ordered.

        :return list(tuple(name:str, value:object, evaluateName:str))
        """
        ret = []
        i = 0
        found_representations = set()
        for key, val in dct.items():
            i += 1
            key_as_str = self.key_to_str(key, fmt)
            if key_as_str not in found_representations:
                found_representations.add(key_as_str)
            else:
                key_as_str = '%s (id: %s)' % (key_as_str, id(key))
                found_representations.add(key_as_str)
            if _does_obj_repr_evaluate_to_obj(key):
                s = self.key_to_str(key)
                eval_key_str = '[%s]' % (s,)
            else:
                eval_key_str = None
            ret.append((key_as_str, val, eval_key_str))
            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                ret.append((TOO_LARGE_ATTR, TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,), None))
                break
        from_default_resolver = defaultResolver.get_contents_debug_adapter_protocol(dct, fmt)
        if from_default_resolver:
            ret = from_default_resolver + ret
        if self.sort_keys:
            ret = sorted(ret, key=lambda tup: sorted_attributes_key(tup[0]))
        ret.append((GENERATED_LEN_ATTR_NAME, len(dct), partial(_apply_evaluate_name, evaluate_name='len(%s)')))
        return ret

    def get_dictionary(self, dct):
        ret = self.init_dict()
        i = 0
        for key, val in dct.items():
            i += 1
            key = '%s (%s)' % (self.key_to_str(key), id(key))
            ret[key] = val
            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                ret[TOO_LARGE_ATTR] = TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,)
                break
        additional_fields = defaultResolver.get_dictionary(dct)
        ret.update(additional_fields)
        ret[GENERATED_LEN_ATTR_NAME] = len(dct)
        return ret