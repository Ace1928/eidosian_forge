import inspect
import io
from typing import (
import click
import click.shell_completion
class ParamMeta:
    empty = inspect.Parameter.empty

    def __init__(self, *, name: str, default: Any=inspect.Parameter.empty, annotation: Any=inspect.Parameter.empty) -> None:
        self.name = name
        self.default = default
        self.annotation = annotation