import abc
import os
import sys
import pathlib
from contextlib import suppress
from typing import Union
def _available_reader(spec):
    with suppress(AttributeError):
        return spec.loader.get_resource_reader(spec.name)