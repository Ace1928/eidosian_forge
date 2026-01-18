import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
Given a type and and a string from the command-line, reconstruct an object. Not
        intended to deal with containers.

        This is intended to replace all calls to `type(string)`, which can cause unexpected
        behavior. As an example, note that the following argparse code will always print
        `True`, because `bool("True") == bool("False") == bool("0") == True`.
        ```
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", type=bool)

        print(parser.parse_args().flag)
        ```
        