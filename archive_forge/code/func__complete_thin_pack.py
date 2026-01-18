import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
def _complete_thin_pack(self, f, path, copier, indexer):
    entries = list(indexer)
    f.seek(0)
    write_pack_header(f, len(entries) + len(indexer.ext_refs()))
    f.flush()
    new_sha = compute_file_sha(f, end_ofs=-20)
    f.seek(0, os.SEEK_CUR)
    for ext_sha in indexer.ext_refs():
        assert len(ext_sha) == 20
        type_num, data = self.get_raw(ext_sha)
        offset = f.tell()
        crc32 = write_pack_object(f, type_num, data, sha=new_sha)
        entries.append((ext_sha, offset, crc32))
    pack_sha = new_sha.digest()
    f.write(pack_sha)
    f.flush()
    entries.sort()
    pack_base_name = posixpath.join(self.pack_dir, 'pack-' + os.fsdecode(iter_sha1((e[0] for e in entries))))
    self.scon.put_object(pack_base_name + '.pack', f)
    filename = pack_base_name + '.idx'
    index_file = BytesIO()
    write_pack_index_v2(index_file, entries, pack_sha)
    self.scon.put_object(filename, index_file)
    f.seek(0)
    pack_data = PackData(filename='', file=f)
    index_file.seek(0)
    pack_index = load_pack_index_file('', index_file)
    serialized_pack_info = pack_info_create(pack_data, pack_index)
    f.close()
    index_file.close()
    pack_info_file = BytesIO(serialized_pack_info)
    filename = pack_base_name + '.info'
    self.scon.put_object(filename, pack_info_file)
    pack_info_file.close()
    final_pack = SwiftPack(pack_base_name, scon=self.scon)
    final_pack.check_length_and_checksum()
    self._add_cached_pack(pack_base_name, final_pack)
    return final_pack