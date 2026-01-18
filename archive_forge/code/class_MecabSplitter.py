import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
class MecabSplitter(BaseSplitter):

    def __init__(self, options: Dict) -> None:
        super().__init__(options)
        self.ctypes_libmecab: Any = None
        self.ctypes_mecab: Any = None
        if not native_module:
            self.init_ctypes(options)
        else:
            self.init_native(options)
        self.dict_encode = options.get('dic_enc', 'utf-8')

    def split(self, input: str) -> List[str]:
        if native_module:
            result = self.native.parse(input)
        else:
            result = self.ctypes_libmecab.mecab_sparse_tostr(self.ctypes_mecab, input.encode(self.dict_encode))
        return result.split(' ')

    def init_native(self, options: Dict) -> None:
        param = '-Owakati'
        dict = options.get('dict')
        if dict:
            param += ' -d %s' % dict
        self.native = MeCab.Tagger(param)

    def init_ctypes(self, options: Dict) -> None:
        import ctypes.util
        lib = options.get('lib')
        if lib is None:
            if sys.platform.startswith('win'):
                libname = 'libmecab.dll'
            else:
                libname = 'mecab'
            libpath = ctypes.util.find_library(libname)
        elif os.path.basename(lib) == lib:
            libpath = ctypes.util.find_library(lib)
        else:
            libpath = None
            if os.path.exists(lib):
                libpath = lib
        if libpath is None:
            raise RuntimeError('MeCab dynamic library is not available')
        param = 'mecab -Owakati'
        dict = options.get('dict')
        if dict:
            param += ' -d %s' % dict
        fs_enc = sys.getfilesystemencoding() or sys.getdefaultencoding()
        self.ctypes_libmecab = ctypes.CDLL(libpath)
        self.ctypes_libmecab.mecab_new2.argtypes = (ctypes.c_char_p,)
        self.ctypes_libmecab.mecab_new2.restype = ctypes.c_void_p
        self.ctypes_libmecab.mecab_sparse_tostr.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
        self.ctypes_libmecab.mecab_sparse_tostr.restype = ctypes.c_char_p
        self.ctypes_mecab = self.ctypes_libmecab.mecab_new2(param.encode(fs_enc))
        if self.ctypes_mecab is None:
            raise SphinxError('mecab initialization failed')

    def __del__(self) -> None:
        if self.ctypes_libmecab:
            self.ctypes_libmecab.mecab_destroy(self.ctypes_mecab)