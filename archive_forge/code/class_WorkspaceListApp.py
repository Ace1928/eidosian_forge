from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any
from jupyter_core.application import JupyterApp
from traitlets import Bool, Unicode
from ._version import __version__
from .config import LabConfig
from .workspaces_handler import WorkspacesManager
class WorkspaceListApp(JupyterApp, LabConfig):
    """An app to list workspaces."""
    version = __version__
    description = "\n    Print all the workspaces available\n\n    If '--json' flag is passed in, a single 'json' object is printed.\n    If '--jsonlines' flag is passed in, 'json' object of each workspace separated by a new line is printed.\n    If nothing is passed in, workspace ids list is printed.\n    "
    flags = dict(jsonlines=({'WorkspaceListApp': {'jsonlines': True}}, 'Produce machine-readable JSON Lines output.'), json=({'WorkspaceListApp': {'json': True}}, 'Produce machine-readable JSON object output.'))
    jsonlines = Bool(False, config=True, help='If True, the output will be a newline-delimited JSON (see https://jsonlines.org/) of objects, one per JupyterLab workspace, each with the details of the relevant workspace')
    json = Bool(False, config=True, help='If True, each line of output will be a JSON object with the details of the workspace.')

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the app."""
        super().initialize(*args, **kwargs)
        self.manager = WorkspacesManager(self.workspaces_dir)

    def start(self) -> None:
        """Start the app."""
        workspaces = self.manager.list_workspaces()
        if self.jsonlines:
            for workspace in workspaces:
                print(json.dumps(workspace))
        elif self.json:
            print(json.dumps(workspaces))
        else:
            for workspace in workspaces:
                print(workspace['metadata']['id'])