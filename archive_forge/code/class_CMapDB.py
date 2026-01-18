import gzip
import logging
import os
import os.path
import pickle as pickle
import struct
import sys
from typing import (
from .encodingdb import name2unicode
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import PSSyntaxError
from .psparser import literal_name
from .utils import choplist
from .utils import nunpack
class CMapDB:
    _cmap_cache: Dict[str, PyCMap] = {}
    _umap_cache: Dict[str, List[PyUnicodeMap]] = {}

    class CMapNotFound(CMapError):
        pass

    @classmethod
    def _load_data(cls, name: str) -> Any:
        name = name.replace('\x00', '')
        filename = '%s.pickle.gz' % name
        log.debug('loading: %r', name)
        cmap_paths = (os.environ.get('CMAP_PATH', '/usr/share/pdfminer/'), os.path.join(os.path.dirname(__file__), 'cmap'))
        for directory in cmap_paths:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                gzfile = gzip.open(path)
                try:
                    return type(str(name), (), pickle.loads(gzfile.read()))
                finally:
                    gzfile.close()
        else:
            raise CMapDB.CMapNotFound(name)

    @classmethod
    def get_cmap(cls, name: str) -> CMapBase:
        if name == 'Identity-H':
            return IdentityCMap(WMode=0)
        elif name == 'Identity-V':
            return IdentityCMap(WMode=1)
        elif name == 'OneByteIdentityH':
            return IdentityCMapByte(WMode=0)
        elif name == 'OneByteIdentityV':
            return IdentityCMapByte(WMode=1)
        try:
            return cls._cmap_cache[name]
        except KeyError:
            pass
        data = cls._load_data(name)
        cls._cmap_cache[name] = cmap = PyCMap(name, data)
        return cmap

    @classmethod
    def get_unicode_map(cls, name: str, vertical: bool=False) -> UnicodeMap:
        try:
            return cls._umap_cache[name][vertical]
        except KeyError:
            pass
        data = cls._load_data('to-unicode-%s' % name)
        cls._umap_cache[name] = [PyUnicodeMap(name, data, v) for v in (False, True)]
        return cls._umap_cache[name][vertical]