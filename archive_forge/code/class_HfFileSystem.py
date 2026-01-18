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
class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        token (`str`, *optional*):
            Authentication token, obtained with [`HfApi.login`] method. Will default to the stored token.

    Usage:

    ```python
    >>> from huggingface_hub import HfFileSystem

    >>> fs = HfFileSystem()

    >>> # List files
    >>> fs.glob("my-username/my-model/*.bin")
    ['my-username/my-model/pytorch_model.bin']
    >>> fs.ls("datasets/my-username/my-dataset", detail=False)
    ['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

    >>> # Read/write files
    >>> with fs.open("my-username/my-model/pytorch_model.bin") as f:
    ...     data = f.read()
    >>> with fs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
    ...     f.write(data)
    ```
    """
    root_marker = ''
    protocol = 'hf'

    def __init__(self, *args, endpoint: Optional[str]=None, token: Optional[str]=None, **storage_options):
        super().__init__(*args, **storage_options)
        self.endpoint = endpoint or ENDPOINT
        self.token = token
        self._api = HfApi(endpoint=endpoint, token=token)
        self._repo_and_revision_exists_cache: Dict[Tuple[str, str, Optional[str]], Tuple[bool, Optional[Exception]]] = {}

    def _repo_and_revision_exist(self, repo_type: str, repo_id: str, revision: Optional[str]) -> Tuple[bool, Optional[Exception]]:
        if (repo_type, repo_id, revision) not in self._repo_and_revision_exists_cache:
            try:
                self._api.repo_info(repo_id, revision=revision, repo_type=repo_type)
            except (RepositoryNotFoundError, HFValidationError) as e:
                self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (False, e)
                self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (False, e)
            except RevisionNotFoundError as e:
                self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (False, e)
                self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (True, None)
            else:
                self._repo_and_revision_exists_cache[repo_type, repo_id, revision] = (True, None)
                self._repo_and_revision_exists_cache[repo_type, repo_id, None] = (True, None)
        return self._repo_and_revision_exists_cache[repo_type, repo_id, revision]

    def resolve_path(self, path: str, revision: Optional[str]=None) -> HfFileSystemResolvedPath:

        def _align_revision_in_path_with_revision(revision_in_path: Optional[str], revision: Optional[str]) -> Optional[str]:
            if revision is not None:
                if revision_in_path is not None and revision_in_path != revision:
                    raise ValueError(f'Revision specified in path ("{revision_in_path}") and in `revision` argument ("{revision}") are not the same.')
            else:
                revision = revision_in_path
            return revision
        path = self._strip_protocol(path)
        if not path:
            raise NotImplementedError('Access to repositories lists is not implemented.')
        elif path.split('/')[0] + '/' in REPO_TYPES_URL_PREFIXES.values():
            if '/' not in path:
                raise NotImplementedError('Access to repositories lists is not implemented.')
            repo_type, path = path.split('/', 1)
            repo_type = REPO_TYPES_MAPPING[repo_type]
        else:
            repo_type = REPO_TYPE_MODEL
        if path.count('/') > 0:
            if '@' in path:
                repo_id, revision_in_path = path.split('@', 1)
                if '/' in revision_in_path:
                    match = SPECIAL_REFS_REVISION_REGEX.search(revision_in_path)
                    if match is not None and revision in (None, match.group()):
                        path_in_repo = SPECIAL_REFS_REVISION_REGEX.sub('', revision_in_path).lstrip('/')
                        revision_in_path = match.group()
                    else:
                        revision_in_path, path_in_repo = revision_in_path.split('/', 1)
                else:
                    path_in_repo = ''
                revision = _align_revision_in_path_with_revision(unquote(revision_in_path), revision)
                repo_and_revision_exist, err = self._repo_and_revision_exist(repo_type, repo_id, revision)
                if not repo_and_revision_exist:
                    _raise_file_not_found(path, err)
            else:
                revision_in_path = None
                repo_id_with_namespace = '/'.join(path.split('/')[:2])
                path_in_repo_with_namespace = '/'.join(path.split('/')[2:])
                repo_id_without_namespace = path.split('/')[0]
                path_in_repo_without_namespace = '/'.join(path.split('/')[1:])
                repo_id = repo_id_with_namespace
                path_in_repo = path_in_repo_with_namespace
                repo_and_revision_exist, err = self._repo_and_revision_exist(repo_type, repo_id, revision)
                if not repo_and_revision_exist:
                    if isinstance(err, (RepositoryNotFoundError, HFValidationError)):
                        repo_id = repo_id_without_namespace
                        path_in_repo = path_in_repo_without_namespace
                        repo_and_revision_exist, _ = self._repo_and_revision_exist(repo_type, repo_id, revision)
                        if not repo_and_revision_exist:
                            _raise_file_not_found(path, err)
                    else:
                        _raise_file_not_found(path, err)
        else:
            repo_id = path
            path_in_repo = ''
            if '@' in path:
                repo_id, revision_in_path = path.split('@', 1)
                revision = _align_revision_in_path_with_revision(unquote(revision_in_path), revision)
            else:
                revision_in_path = None
            repo_and_revision_exist, _ = self._repo_and_revision_exist(repo_type, repo_id, revision)
            if not repo_and_revision_exist:
                raise NotImplementedError('Access to repositories lists is not implemented.')
        revision = revision if revision is not None else DEFAULT_REVISION
        return HfFileSystemResolvedPath(repo_type, repo_id, revision, path_in_repo, _raw_revision=revision_in_path)

    def invalidate_cache(self, path: Optional[str]=None) -> None:
        if not path:
            self.dircache.clear()
            self._repo_and_revision_exists_cache.clear()
        else:
            path = self.resolve_path(path).unresolve()
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

    def _open(self, path: str, mode: str='rb', revision: Optional[str]=None, block_size: Optional[int]=None, **kwargs) -> 'HfFileSystemFile':
        if 'a' in mode:
            raise NotImplementedError('Appending to remote files is not yet supported.')
        if block_size == 0:
            return HfFileSystemStreamFile(self, path, mode=mode, revision=revision, block_size=block_size, **kwargs)
        else:
            return HfFileSystemFile(self, path, mode=mode, revision=revision, block_size=block_size, **kwargs)

    def _rm(self, path: str, revision: Optional[str]=None, **kwargs) -> None:
        resolved_path = self.resolve_path(path, revision=revision)
        self._api.delete_file(path_in_repo=resolved_path.path_in_repo, repo_id=resolved_path.repo_id, token=self.token, repo_type=resolved_path.repo_type, revision=resolved_path.revision, commit_message=kwargs.get('commit_message'), commit_description=kwargs.get('commit_description'))
        self.invalidate_cache(path=resolved_path.unresolve())

    def rm(self, path: str, recursive: bool=False, maxdepth: Optional[int]=None, revision: Optional[str]=None, **kwargs) -> None:
        resolved_path = self.resolve_path(path, revision=revision)
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth, revision=revision)
        paths_in_repo = [self.resolve_path(path).path_in_repo for path in paths if not self.isdir(path)]
        operations = [CommitOperationDelete(path_in_repo=path_in_repo) for path_in_repo in paths_in_repo]
        commit_message = f'Delete {path} '
        commit_message += 'recursively ' if recursive else ''
        commit_message += f'up to depth {maxdepth} ' if maxdepth is not None else ''
        self._api.create_commit(repo_id=resolved_path.repo_id, repo_type=resolved_path.repo_type, token=self.token, operations=operations, revision=resolved_path.revision, commit_message=kwargs.get('commit_message', commit_message), commit_description=kwargs.get('commit_description'))
        self.invalidate_cache(path=resolved_path.unresolve())

    def ls(self, path: str, detail: bool=True, refresh: bool=False, revision: Optional[str]=None, **kwargs) -> List[Union[str, Dict[str, Any]]]:
        """List the contents of a directory."""
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        kwargs = {'expand_info': detail, **kwargs}
        try:
            out = self._ls_tree(path, refresh=refresh, revision=revision, **kwargs)
        except EntryNotFoundError:
            if not resolved_path.path_in_repo:
                _raise_file_not_found(path, None)
            out = self._ls_tree(self._parent(path), refresh=refresh, revision=revision, **kwargs)
            out = [o for o in out if o['name'] == path]
            if len(out) == 0:
                _raise_file_not_found(path, None)
        return out if detail else [o['name'] for o in out]

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

    def glob(self, path, **kwargs):
        kwargs = {'expand_info': kwargs.get('detail', False), **kwargs}
        path = self.resolve_path(path, revision=kwargs.get('revision')).unresolve()
        return super().glob(path, **kwargs)

    def find(self, path: str, maxdepth: Optional[int]=None, withdirs: bool=False, detail: bool=False, refresh: bool=False, revision: Optional[str]=None, **kwargs) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        if maxdepth:
            return super().find(path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, refresh=refresh, revision=revision, **kwargs)
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        kwargs = {'expand_info': detail, **kwargs}
        try:
            out = self._ls_tree(path, recursive=True, refresh=refresh, revision=resolved_path.revision, **kwargs)
        except EntryNotFoundError:
            if self.info(path, revision=revision, **kwargs)['type'] == 'file':
                out = {path: {}}
            else:
                out = {}
        else:
            if not withdirs:
                out = [o for o in out if o['type'] != 'directory']
            else:
                path_info = self.info(path, revision=resolved_path.revision, **kwargs)
                out = [path_info] + out if path_info['type'] == 'directory' else out
            out = {o['name']: o for o in out}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    def cp_file(self, path1: str, path2: str, revision: Optional[str]=None, **kwargs) -> None:
        resolved_path1 = self.resolve_path(path1, revision=revision)
        resolved_path2 = self.resolve_path(path2, revision=revision)
        same_repo = resolved_path1.repo_type == resolved_path2.repo_type and resolved_path1.repo_id == resolved_path2.repo_id
        if same_repo:
            commit_message = f'Copy {path1} to {path2}'
            self._api.create_commit(repo_id=resolved_path1.repo_id, repo_type=resolved_path1.repo_type, revision=resolved_path2.revision, commit_message=kwargs.get('commit_message', commit_message), commit_description=kwargs.get('commit_description', ''), operations=[CommitOperationCopy(src_path_in_repo=resolved_path1.path_in_repo, path_in_repo=resolved_path2.path_in_repo, src_revision=resolved_path1.revision)])
        else:
            with self.open(path1, 'rb', revision=resolved_path1.revision) as f:
                content = f.read()
            commit_message = f'Copy {path1} to {path2}'
            self._api.upload_file(path_or_fileobj=content, path_in_repo=resolved_path2.path_in_repo, repo_id=resolved_path2.repo_id, token=self.token, repo_type=resolved_path2.repo_type, revision=resolved_path2.revision, commit_message=kwargs.get('commit_message', commit_message), commit_description=kwargs.get('commit_description'))
        self.invalidate_cache(path=resolved_path1.unresolve())
        self.invalidate_cache(path=resolved_path2.unresolve())

    def modified(self, path: str, **kwargs) -> datetime:
        info = self.info(path, **kwargs)
        return info['last_commit']['date']

    def info(self, path: str, refresh: bool=False, revision: Optional[str]=None, **kwargs) -> Dict[str, Any]:
        resolved_path = self.resolve_path(path, revision=revision)
        path = resolved_path.unresolve()
        expand_info = kwargs.get('expand_info', True)
        if not resolved_path.path_in_repo:
            out = {'name': path, 'size': 0, 'type': 'directory'}
            if expand_info:
                last_commit = self._api.list_repo_commits(resolved_path.repo_id, repo_type=resolved_path.repo_type, revision=resolved_path.revision)[-1]
                out = {**out, 'tree_id': None, 'last_commit': LastCommitInfo(oid=last_commit.commit_id, title=last_commit.title, date=last_commit.created_at)}
        else:
            out = None
            parent_path = self._parent(path)
            if parent_path in self.dircache:
                out1 = [o for o in self.dircache[parent_path] if o['name'] == path]
                if not out1:
                    _raise_file_not_found(path, None)
                out = out1[0]
            if refresh or out is None or (expand_info and out and (out['last_commit'] is None)):
                paths_info = self._api.get_paths_info(resolved_path.repo_id, resolved_path.path_in_repo, expand=expand_info, revision=resolved_path.revision, repo_type=resolved_path.repo_type)
                if not paths_info:
                    _raise_file_not_found(path, None)
                path_info = paths_info[0]
                root_path = HfFileSystemResolvedPath(resolved_path.repo_type, resolved_path.repo_id, resolved_path.revision, path_in_repo='', _raw_revision=resolved_path._raw_revision).unresolve()
                if isinstance(path_info, RepoFile):
                    out = {'name': root_path + '/' + path_info.path, 'size': path_info.size, 'type': 'file', 'blob_id': path_info.blob_id, 'lfs': path_info.lfs, 'last_commit': path_info.last_commit, 'security': path_info.security}
                else:
                    out = {'name': root_path + '/' + path_info.path, 'size': 0, 'type': 'directory', 'tree_id': path_info.tree_id, 'last_commit': path_info.last_commit}
                if not expand_info:
                    out = {k: out[k] for k in ['name', 'size', 'type']}
        assert out is not None
        return copy.deepcopy(out)

    def exists(self, path, **kwargs):
        """Is there a file at the given path"""
        try:
            self.info(path, **{**kwargs, 'expand_info': False})
            return True
        except:
            return False

    def isdir(self, path):
        """Is this entry directory-like?"""
        try:
            return self.info(path, expand_info=False)['type'] == 'directory'
        except OSError:
            return False

    def isfile(self, path):
        """Is this entry file-like?"""
        try:
            return self.info(path, expand_info=False)['type'] == 'file'
        except:
            return False

    def url(self, path: str) -> str:
        """Get the HTTP URL of the given path"""
        resolved_path = self.resolve_path(path)
        url = hf_hub_url(resolved_path.repo_id, resolved_path.path_in_repo, repo_type=resolved_path.repo_type, revision=resolved_path.revision, endpoint=self.endpoint)
        if self.isdir(path):
            url = url.replace('/resolve/', '/tree/', 1)
        return url

    @property
    def transaction(self):
        """A context within which files are committed together upon exit

        Requires the file class to implement `.commit()` and `.discard()`
        for the normal and exception cases.
        """
        raise NotImplementedError('Transactional commits are not supported.')

    def start_transaction(self):
        """Begin write transaction for deferring files, non-context version"""
        raise NotImplementedError('Transactional commits are not supported.')