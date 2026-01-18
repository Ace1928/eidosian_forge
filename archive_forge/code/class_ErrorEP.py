from __future__ import annotations
import importlib
import json
import pathlib
import platform
import sys
import click
import pytest
import yaml
from click.testing import CliRunner
import dask
import dask.cli
from dask._compatibility import importlib_metadata
class ErrorEP:

    @property
    def name(self):
        return 'foo'

    def load(self):
        raise ImportError('Entrypoint could not be imported')