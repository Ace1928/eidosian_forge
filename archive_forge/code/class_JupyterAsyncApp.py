from __future__ import annotations
import logging
import os
import sys
import typing as t
from copy import deepcopy
from pathlib import Path
from shutil import which
from traitlets import Bool, List, Unicode, observe
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound
from .paths import (
from .utils import ensure_dir_exists, ensure_event_loop
class JupyterAsyncApp(JupyterApp):
    """A Jupyter application that runs on an asyncio loop."""
    name = 'jupyter_async'
    description = 'An Async Jupyter Application'
    _prefer_selector_loop = False

    async def initialize_async(self, argv: t.Any=None) -> None:
        """Initialize the application asynchronoously."""

    async def start_async(self) -> None:
        """Run the application in an event loop."""

    @classmethod
    async def _launch_instance(cls, argv: t.Any=None, **kwargs: t.Any) -> None:
        app = cls.instance(**kwargs)
        app.initialize(argv)
        await app.initialize_async(argv)
        await app.start_async()

    @classmethod
    def launch_instance(cls, argv: t.Any=None, **kwargs: t.Any) -> None:
        """Launch an instance of an async Jupyter Application"""
        loop = ensure_event_loop(cls._prefer_selector_loop)
        coro = cls._launch_instance(argv, **kwargs)
        loop.run_until_complete(coro)
        loop.close()