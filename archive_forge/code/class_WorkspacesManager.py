from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import urllib
from pathlib import Path
from typing import Any
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from traitlets.config import LoggingConfigurable
class WorkspacesManager(LoggingConfigurable):
    """A manager for workspaces."""

    def __init__(self, path: str) -> None:
        """Initialize a workspaces manager with content in ``path``."""
        super()
        if not path:
            msg = 'Workspaces directory is not set'
            raise ValueError(msg)
        self.workspaces_dir = Path(path)

    def delete(self, space_name: str) -> None:
        """Remove a workspace ``space_name``."""
        slug = slugify(space_name)
        workspace_path = self.workspaces_dir / (slug + WORKSPACE_EXTENSION)
        if not workspace_path.exists():
            msg = f'Workspace {space_name!r} ({slug!r}) not found'
            raise FileNotFoundError(msg)
        workspace_path.unlink()

    def list_workspaces(self) -> list:
        """List all available workspaces."""
        prefix = slugify('', sign=False)
        return _list_workspaces(self.workspaces_dir, prefix)

    def load(self, space_name: str) -> dict:
        """Load the workspace ``space_name``."""
        slug = slugify(space_name)
        workspace_path = self.workspaces_dir / (slug + WORKSPACE_EXTENSION)
        if workspace_path.exists():
            return _load_with_file_times(workspace_path)
        _id = space_name if space_name.startswith('/') else '/' + space_name
        return dict(data=dict(), metadata=dict(id=_id))

    def save(self, space_name: str, raw: str) -> Path:
        """Save the ``raw`` data as workspace ``space_name``."""
        if not self.workspaces_dir.exists():
            self.workspaces_dir.mkdir(parents=True)
        workspace = {}
        try:
            decoder = json.JSONDecoder()
            workspace = decoder.decode(raw)
        except Exception as e:
            raise ValueError(str(e)) from e
        metadata_id = workspace['metadata']['id']
        metadata_id = metadata_id if metadata_id.startswith('/') else '/' + metadata_id
        metadata_id = urllib.parse.unquote(metadata_id)
        if metadata_id != '/' + space_name:
            message = f'Workspace metadata ID mismatch: expected {space_name!r} got {metadata_id!r}'
            raise ValueError(message)
        slug = slugify(space_name)
        workspace_path = self.workspaces_dir / (slug + WORKSPACE_EXTENSION)
        workspace_path.write_text(raw, encoding='utf-8')
        return workspace_path