from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
class CallSignatureCache(object):

    def __init__(self):
        self.cache = {}

    def add(self, signature):
        filename, name, args_type = get_signature_info(signature)
        calls_from_file = self.cache.setdefault(filename, {})
        name_calls = calls_from_file.setdefault(name, {})
        name_calls[args_type] = None

    def is_in_cache(self, signature):
        filename, name, args_type = get_signature_info(signature)
        if args_type in self.cache.get(filename, {}).get(name, {}):
            return True
        return False