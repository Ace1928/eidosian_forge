from __future__ import annotations
import typing
from copy import copy, deepcopy
from typing import Iterable, List, cast, overload
import pandas as pd
from ._utils import array_kind, check_required_aesthetics, ninteraction
from .exceptions import PlotnineError
from .mapping.aes import NO_GROUP, SCALED_AESTHETICS, aes
from .mapping.evaluation import evaluate, stage
def _make_layer_environments(self, plot_environment: Environment):
    """
        Create the aesthetic mappings to be used by this layer

        Parameters
        ----------
        plot_environment :
            Namespace in which to execute aesthetic expressions.
        """
    self.geom.environment = plot_environment
    self.stat.environment = plot_environment