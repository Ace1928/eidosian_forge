import os
import shutil
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode
from jupyter_server import _tz as tz
from .checkpoints import (
from .fileio import AsyncFileManagerMixin, FileManagerMixin
class GenericFileCheckpoints(GenericCheckpointsMixin, FileCheckpoints):
    """
    Local filesystem Checkpoints that works with any conforming
    ContentsManager.
    """

    def create_file_checkpoint(self, content, format, path):
        """Create a checkpoint from the current content of a file."""
        path = path.strip('/')
        checkpoint_id = 'checkpoint'
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug('creating checkpoint for %s', path)
        with self.perm_to_403():
            self._save_file(os_checkpoint_path, content, format=format)
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def create_notebook_checkpoint(self, nb, path):
        """Create a checkpoint from the current content of a notebook."""
        path = path.strip('/')
        checkpoint_id = 'checkpoint'
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug('creating checkpoint for %s', path)
        with self.perm_to_403():
            self._save_notebook(os_checkpoint_path, nb)
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """Get a checkpoint for a notebook."""
        path = path.strip('/')
        self.log.info('restoring %s from checkpoint %s', path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)
        return {'type': 'notebook', 'content': self._read_notebook(os_checkpoint_path, as_version=4)}

    def get_file_checkpoint(self, checkpoint_id, path):
        """Get a checkpoint for a file."""
        path = path.strip('/')
        self.log.info('restoring %s from checkpoint %s', path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)
        content, format = self._read_file(os_checkpoint_path, format=None)
        return {'type': 'file', 'content': content, 'format': format}