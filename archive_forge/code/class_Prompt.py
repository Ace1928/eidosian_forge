import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
@dataclass
class Prompt:
    """Represents a prompt function.

    We return a `Prompt` class instead of a simple function so the
    template defined in prompt functions can be accessed.

    """
    template: str
    signature: inspect.Signature

    def __post_init__(self):
        self.parameters: List[str] = list(self.signature.parameters.keys())

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template.

        Returns
        -------
        The rendered template as a Python ``str``.

        """
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(self.template, **bound_arguments.arguments)

    def __str__(self):
        return self.template