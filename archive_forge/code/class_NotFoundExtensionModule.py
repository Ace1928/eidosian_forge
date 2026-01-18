from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
class NotFoundExtensionModule(NewExtensionModule):
    """Class for modern modules

    provides the found method.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.INFO = ModuleInfo(name)

    @staticmethod
    def found() -> bool:
        return False