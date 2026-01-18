import logging
import sys
from typing import Any, List
from .threading import run_once
from importlib.metadata import EntryPoint, entry_points
Load dependencies and functions of a given entrypoint. For any
    given entrypoint name, it will be loaded only once in one process.

    :param name: the name of the entrypoint

    .. admonition:: Example

        Assume in ``setup.py``, you have:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And this is how you load ``my.plugins``:

        .. code-block:: python

            from triad.utils.entrypoints import load_entry_point

            load_entry_point("my.plugins")
    