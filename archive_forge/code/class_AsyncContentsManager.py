from __future__ import annotations
import itertools
import json
import os
import re
import typing as t
import warnings
from fnmatch import fnmatch
from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item
from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints
class AsyncContentsManager(ContentsManager):
    """Base class for serving files and directories asynchronously."""
    checkpoints_class = Type(AsyncCheckpoints, config=True)
    checkpoints = Instance(AsyncCheckpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default('checkpoints')
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default('checkpoints_kwargs')
    def _default_checkpoints_kwargs(self):
        return {'parent': self, 'log': self.log}

    async def dir_exists(self, path):
        """Does a directory exist at the given path?

        Like os.path.isdir

        Override this method in subclasses.

        Parameters
        ----------
        path : str
            The path to check

        Returns
        -------
        exists : bool
            Whether the path does indeed exist.
        """
        raise NotImplementedError

    async def is_hidden(self, path):
        """Is path a hidden directory or file?

        Parameters
        ----------
        path : str
            The path to check. This is an API path (`/` separated,
            relative to root dir).

        Returns
        -------
        hidden : bool
            Whether the path is hidden.

        """
        raise NotImplementedError

    async def file_exists(self, path=''):
        """Does a file exist at the given path?

        Like os.path.isfile

        Override this method in subclasses.

        Parameters
        ----------
        path : str
            The API path of a file to check for.

        Returns
        -------
        exists : bool
            Whether the file exists.
        """
        raise NotImplementedError

    async def exists(self, path):
        """Does a file or directory exist at the given path?

        Like os.path.exists

        Parameters
        ----------
        path : str
            The API path of a file or directory to check for.

        Returns
        -------
        exists : bool
            Whether the target exists.
        """
        return await ensure_async(self.file_exists(path)) or await ensure_async(self.dir_exists(path))

    async def get(self, path, content=True, type=None, format=None, require_hash=False):
        """Get a file or directory model.

        Parameters
        ----------
        require_hash : bool
            Whether the file hash must be returned or not.

        *Changed in version 2.11*: The *require_hash* parameter was added.
        """
        raise NotImplementedError

    async def save(self, model, path):
        """
        Save a file or directory model to path.

        Should return the saved model with no content.  Save implementations
        should call self.run_pre_save_hook(model=model, path=path) prior to
        writing any data.
        """
        raise NotImplementedError

    async def delete_file(self, path):
        """Delete the file or directory at path."""
        raise NotImplementedError

    async def rename_file(self, old_path, new_path):
        """Rename a file or directory."""
        raise NotImplementedError

    async def delete(self, path):
        """Delete a file/directory and any associated checkpoints."""
        path = path.strip('/')
        if not path:
            raise HTTPError(400, "Can't delete root")
        await self.delete_file(path)
        await self.checkpoints.delete_all_checkpoints(path)
        self.emit(data={'action': 'delete', 'path': path})

    async def rename(self, old_path, new_path):
        """Rename a file and any checkpoints associated with that file."""
        await self.rename_file(old_path, new_path)
        await self.checkpoints.rename_all_checkpoints(old_path, new_path)
        self.emit(data={'action': 'rename', 'path': new_path, 'source_path': old_path})

    async def update(self, model, path):
        """Update the file's path

        For use in PATCH requests, to enable renaming a file without
        re-uploading its contents. Only used for renaming at the moment.
        """
        path = path.strip('/')
        new_path = model.get('path', path).strip('/')
        if path != new_path:
            await self.rename(path, new_path)
        model = await self.get(new_path, content=False)
        return model

    async def increment_filename(self, filename, path='', insert=''):
        """Increment a filename until it is unique.

        Parameters
        ----------
        filename : unicode
            The name of a file, including extension
        path : unicode
            The API path of the target's directory
        insert : unicode
            The characters to insert after the base filename

        Returns
        -------
        name : unicode
            A filename that is unique, based on the input filename.
        """
        path = path.strip('/')
        basename, dot, ext = filename.rpartition('.')
        if ext != 'ipynb':
            basename, dot, ext = filename.partition('.')
        suffix = dot + ext
        for i in itertools.count():
            insert_i = f'{insert}{i}' if i else ''
            name = f'{basename}{insert_i}{suffix}'
            file_exists = await ensure_async(self.exists(f'{path}/{name}'))
            if not file_exists:
                break
        return name

    async def new_untitled(self, path='', type='', ext=''):
        """Create a new untitled file or directory in path

        path must be a directory

        File extension can be specified.

        Use `new` to create files with a fully specified path (including filename).
        """
        path = path.strip('/')
        dir_exists = await ensure_async(self.dir_exists(path))
        if not dir_exists:
            raise HTTPError(404, 'No such directory: %s' % path)
        model = {}
        if type:
            model['type'] = type
        if ext == '.ipynb':
            model.setdefault('type', 'notebook')
        else:
            model.setdefault('type', 'file')
        insert = ''
        if model['type'] == 'directory':
            untitled = self.untitled_directory
            insert = ' '
        elif model['type'] == 'notebook':
            untitled = self.untitled_notebook
            ext = '.ipynb'
        elif model['type'] == 'file':
            untitled = self.untitled_file
        else:
            raise HTTPError(400, 'Unexpected model type: %r' % model['type'])
        name = await self.increment_filename(untitled + ext, path, insert=insert)
        path = f'{path}/{name}'
        return await self.new(model, path)

    async def new(self, model=None, path=''):
        """Create a new file or directory and return its model with no content.

        To create a new untitled entity in a directory, use `new_untitled`.
        """
        path = path.strip('/')
        if model is None:
            model = {}
        if path.endswith('.ipynb'):
            model.setdefault('type', 'notebook')
        else:
            model.setdefault('type', 'file')
        if 'content' not in model and model['type'] != 'directory':
            if model['type'] == 'notebook':
                model['content'] = new_notebook()
                model['format'] = 'json'
            else:
                model['content'] = ''
                model['type'] = 'file'
                model['format'] = 'text'
        model = await self.save(model, path)
        return model

    async def copy(self, from_path, to_path=None):
        """Copy an existing file and return its new model.

        If to_path not specified, it will be the parent directory of from_path.
        If to_path is a directory, filename will increment `from_path-Copy#.ext`.
        Considering multi-part extensions, the Copy# part will be placed before the first dot for all the extensions except `ipynb`.
        For easier manual searching in case of notebooks, the Copy# part will be placed before the last dot.

        from_path must be a full path to a file.
        """
        path = from_path.strip('/')
        if to_path is not None:
            to_path = to_path.strip('/')
        if '/' in path:
            from_dir, from_name = path.rsplit('/', 1)
        else:
            from_dir = ''
            from_name = path
        model = await self.get(path)
        model.pop('path', None)
        model.pop('name', None)
        if model['type'] == 'directory':
            raise HTTPError(400, "Can't copy directories")
        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if await ensure_async(self.dir_exists(to_path)):
            name = copy_pat.sub('.', from_name)
            to_name = await self.increment_filename(name, to_path, insert='-Copy')
            to_path = f'{to_path}/{to_name}'
        elif is_destination_specified:
            if '/' in to_path:
                to_dir, to_name = to_path.rsplit('/', 1)
                if not await ensure_async(self.dir_exists(to_dir)):
                    raise HTTPError(404, 'No such parent directory: %s to copy file in' % to_dir)
        else:
            raise HTTPError(404, 'No such directory: %s' % to_path)
        model = await self.save(model, to_path)
        self.emit(data={'action': 'copy', 'path': to_path, 'source_path': from_path})
        return model

    async def trust_notebook(self, path):
        """Explicitly trust a notebook

        Parameters
        ----------
        path : str
            The path of a notebook
        """
        model = await self.get(path)
        nb = model['content']
        self.log.warning('Trusting notebook %s', path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    async def create_checkpoint(self, path):
        """Create a checkpoint."""
        return await self.checkpoints.create_checkpoint(self, path)

    async def restore_checkpoint(self, checkpoint_id, path):
        """
        Restore a checkpoint.
        """
        await self.checkpoints.restore_checkpoint(self, checkpoint_id, path)

    async def list_checkpoints(self, path):
        """List the checkpoints for a path."""
        return await self.checkpoints.list_checkpoints(path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """Delete a checkpoint for a path by id."""
        return await self.checkpoints.delete_checkpoint(checkpoint_id, path)