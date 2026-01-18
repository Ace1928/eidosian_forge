from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
def _analyze_paths(file_list, fs, root=False):
    """Consolidate list of file-paths into parquet relative paths

    Note: This function was mostly copied from dask/fastparquet to
    use in both `FastParquetEngine` and `ArrowEngine`."""

    def _join_path(*path):

        def _scrub(i, p):
            p = p.replace(fs.sep, '/')
            if p == '':
                return '.'
            if p[-1] == '/':
                p = p[:-1]
            if i > 0 and p[0] == '/':
                p = p[1:]
            return p
        abs_prefix = ''
        if path and path[0]:
            if path[0][0] == '/':
                abs_prefix = '/'
                path = list(path)
                path[0] = path[0][1:]
            elif fs.sep == '\\' and path[0][1:].startswith(':/'):
                abs_prefix = path[0][0:3]
                path = list(path)
                path[0] = path[0][3:]
        _scrubbed = []
        for i, p in enumerate(path):
            _scrubbed.extend(_scrub(i, p).split('/'))
        simpler = []
        for s in _scrubbed:
            if s == '.':
                pass
            elif s == '..':
                if simpler:
                    if simpler[-1] == '..':
                        simpler.append(s)
                    else:
                        simpler.pop()
                elif abs_prefix:
                    raise Exception('can not get parent of root')
                else:
                    simpler.append(s)
            else:
                simpler.append(s)
        if not simpler:
            if abs_prefix:
                joined = abs_prefix
            else:
                joined = '.'
        else:
            joined = abs_prefix + '/'.join(simpler)
        return joined
    path_parts_list = [_join_path(fn).split('/') for fn in file_list]
    if root is False:
        basepath = path_parts_list[0][:-1]
        for path_parts in path_parts_list:
            j = len(path_parts) - 1
            for k, (base_part, path_part) in enumerate(zip(basepath, path_parts)):
                if base_part != path_part:
                    j = k
                    break
            basepath = basepath[:j]
        l = len(basepath)
    else:
        basepath = _join_path(root).split('/')
        l = len(basepath)
        assert all((p[:l] == basepath for p in path_parts_list)), 'All paths must begin with the given root'
    out_list = []
    for path_parts in path_parts_list:
        out_list.append('/'.join(path_parts[l:]))
    return ('/'.join(basepath), out_list)