import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
class KernelTemplate:
    """
    Base class for defining kernel templates.

    Children classes: TritonTemplate, CUDATemplate
    """

    @staticmethod
    def _template_from_string(source):
        env = jinja2_env()
        if env is not None:
            return env.from_string(source)
        return None

    @staticmethod
    def _fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype

        def get_dtype(name):
            if name == fake_out.get_name():
                return fake_out.get_dtype()
            return _get_dtype_real(name)
        return get_dtype

    def __init__(self, name: str):
        self.name = name

    def maybe_append_choice(self, choices, **kwargs):
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
        try:
            choices.append(self.generate(**kwargs))
        except NotImplementedError:
            pass

    def generate(self, **kwargs) -> ChoiceCaller:
        """
        Generates a ChoiceCaller instance from the given arguments.
        """
        raise NotImplementedError()