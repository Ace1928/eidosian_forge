from __future__ import annotations
import io
import typing as ty
from copy import copy
from .openers import ImageOpener
class FileHolderError(Exception):
    pass