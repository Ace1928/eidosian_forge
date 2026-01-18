from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def _open_browser_with_webbrowser(url: str) -> None:
    import webbrowser
    webbrowser.open(url)