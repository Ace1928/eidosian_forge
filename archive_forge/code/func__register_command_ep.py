from __future__ import annotations
import pathlib
import warnings
from functools import reduce
from typing import Any
import click
import yaml
import dask
from dask import __version__
from dask._compatibility import importlib_metadata
def _register_command_ep(interface, entry_point):
    """Add `entry_point` command to `interface`.

    Parameters
    ----------
    interface : click.Command or click.Group
        The click interface to augment with `entry_point`.
    entry_point : importlib.metadata.EntryPoint
        The entry point which loads to a ``click.Command`` or
        ``click.Group`` instance to be added as a sub-command or
        sub-group in `interface`.

    """
    try:
        command = entry_point.load()
    except Exception as e:
        warnings.warn(f"While registering the command with name '{entry_point.name}', an exception occurred; {e}.")
        return
    if not isinstance(command, (click.Command, click.Group)):
        warnings.warn(f"entry points in 'dask_cli' must be instances of click.Command or click.Group, not {type(command)}.")
        return
    if command.name in interface.commands:
        warnings.warn(f"While registering the command with name '{command.name}', an existing command or group; the original has been overwritten.")
    interface.add_command(command)