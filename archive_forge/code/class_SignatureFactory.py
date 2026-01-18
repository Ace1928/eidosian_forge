from _pydev_bundle import pydev_log
import os
from _pydevd_bundle.pydevd_comm import CMD_SIGNATURE_CALL_TRACE, NetCommand
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
class SignatureFactory(object):

    def __init__(self):
        self._caller_cache = {}
        self.cache = CallSignatureCache()

    def create_signature(self, frame, filename, with_args=True):
        try:
            _, modulename, funcname = self.file_module_function_of(frame)
            signature = Signature(filename, funcname)
            if with_args:
                signature.set_args(frame, recursive=True)
            return signature
        except:
            pydev_log.exception()

    def file_module_function_of(self, frame):
        code = frame.f_code
        filename = code.co_filename
        if filename:
            modulename = _modname(filename)
        else:
            modulename = None
        funcname = code.co_name
        clsname = None
        if code in self._caller_cache:
            if self._caller_cache[code] is not None:
                clsname = self._caller_cache[code]
        else:
            self._caller_cache[code] = None
            clsname = get_clsname_for_code(code, frame)
            if clsname is not None:
                self._caller_cache[code] = clsname
        if clsname is not None:
            funcname = '%s.%s' % (clsname, funcname)
        return (filename, modulename, funcname)