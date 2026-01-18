from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
def _verify_schema(s):
    assert isinstance(s, dict), 'Schema must be dictionary'
    for field in ['name', 'type', 'fields']:
        assert field in s, "Schema missing '%s' field" % field
    assert s['type'] == 'record', "Schema must be of type 'record'"
    assert isinstance(s['fields'], list), 'Fields entry must be a list'
    for f in s['fields']:
        assert 'name' in f and 'type' in f, 'Field spec incomplete: %s' % f