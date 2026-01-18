import copy
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from urllib.parse import quote, unquote
import fsspec
from requests import Response
from ._commit_api import CommitOperationCopy, CommitOperationDelete
from .constants import DEFAULT_REVISION, ENDPOINT, REPO_TYPE_MODEL, REPO_TYPES_MAPPING, REPO_TYPES_URL_PREFIXES
from .file_download import hf_hub_url
from .hf_api import HfApi, LastCommitInfo, RepoFile
from .utils import (
def _ls_tree(self, path: str, recursive: bool=False, refresh: bool=False, revision: Optional[str]=None, expand_info: bool=True):
    resolved_path = self.resolve_path(path, revision=revision)
    path = resolved_path.unresolve()
    root_path = HfFileSystemResolvedPath(resolved_path.repo_type, resolved_path.repo_id, resolved_path.revision, path_in_repo='', _raw_revision=resolved_path._raw_revision).unresolve()
    out = []
    if path in self.dircache and (not refresh):
        cached_path_infos = self.dircache[path]
        out.extend(cached_path_infos)
        dirs_not_in_dircache = []
        if recursive:
            dirs_to_visit = deque([path_info for path_info in cached_path_infos if path_info['type'] == 'directory'])
            while dirs_to_visit:
                dir_info = dirs_to_visit.popleft()
                if dir_info['name'] not in self.dircache:
                    dirs_not_in_dircache.append(dir_info['name'])
                else:
                    cached_path_infos = self.dircache[dir_info['name']]
                    out.extend(cached_path_infos)
                    dirs_to_visit.extend([path_info for path_info in cached_path_infos if path_info['type'] == 'directory'])
        dirs_not_expanded = []
        if expand_info:
            dirs_not_expanded = [self._parent(o['name']) for o in out if o['last_commit'] is None]
        if recursive and dirs_not_in_dircache or (expand_info and dirs_not_expanded):
            common_prefix = os.path.commonprefix(dirs_not_in_dircache + dirs_not_expanded)
            common_path = common_prefix.rstrip('/') if common_prefix.endswith('/') or common_prefix == root_path or common_prefix in chain(dirs_not_in_dircache, dirs_not_expanded) else self._parent(common_prefix)
            out = [o for o in out if not o['name'].startswith(common_path + '/')]
            for cached_path in self.dircache:
                if cached_path.startswith(common_path + '/'):
                    self.dircache.pop(cached_path, None)
            self.dircache.pop(common_path, None)
            out.extend(self._ls_tree(common_path, recursive=recursive, refresh=True, revision=revision, expand_info=expand_info))
    else:
        tree = self._api.list_repo_tree(resolved_path.repo_id, resolved_path.path_in_repo, recursive=recursive, expand=expand_info, revision=resolved_path.revision, repo_type=resolved_path.repo_type)
        for path_info in tree:
            if isinstance(path_info, RepoFile):
                cache_path_info = {'name': root_path + '/' + path_info.path, 'size': path_info.size, 'type': 'file', 'blob_id': path_info.blob_id, 'lfs': path_info.lfs, 'last_commit': path_info.last_commit, 'security': path_info.security}
            else:
                cache_path_info = {'name': root_path + '/' + path_info.path, 'size': 0, 'type': 'directory', 'tree_id': path_info.tree_id, 'last_commit': path_info.last_commit}
            parent_path = self._parent(cache_path_info['name'])
            self.dircache.setdefault(parent_path, []).append(cache_path_info)
            out.append(cache_path_info)
    return copy.deepcopy(out)