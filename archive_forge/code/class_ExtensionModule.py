from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
class ExtensionModule(NewExtensionModule):

    def __init__(self, interpreter: 'Interpreter') -> None:
        super().__init__()
        self.interpreter = interpreter