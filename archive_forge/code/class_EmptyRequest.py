import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
class EmptyRequest:

    def get(self, name, default=None):
        return Bunch(**{method: dict() for method in METHODS})

    def __getitem__(self, name):
        return Bunch(**{method: dict() for method in METHODS})

    def __getattr__(self, name):
        return Bunch(**{method: dict() for method in METHODS})