import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
class ModifyCheckpointsHandler(ContentsAPIHandler):
    """A checkpoints modification handler."""

    @web.authenticated
    @authorized
    async def post(self, path, checkpoint_id):
        """post restores a file from a checkpoint"""
        cm = self.contents_manager
        await ensure_async(cm.restore_checkpoint(checkpoint_id, path))
        self.set_status(204)
        self.finish()

    @web.authenticated
    @authorized
    async def delete(self, path, checkpoint_id):
        """delete clears a checkpoint for a given file"""
        cm = self.contents_manager
        await ensure_async(cm.delete_checkpoint(checkpoint_id, path))
        self.set_status(204)
        self.finish()