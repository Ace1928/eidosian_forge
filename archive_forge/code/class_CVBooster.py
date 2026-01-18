import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from . import callback
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _InnerPredictor,
from .compat import SKLEARN_INSTALLED, _LGBMBaseCrossValidator, _LGBMGroupKFold, _LGBMStratifiedKFold
class CVBooster:
    """CVBooster in LightGBM.

    Auxiliary data structure to hold and redirect all boosters of ``cv()`` function.
    This class has the same methods as Booster class.
    All method calls, except for the following methods, are actually performed for underlying Boosters and
    then all returned results are returned in a list.

    - ``model_from_string()``
    - ``model_to_string()``
    - ``save_model()``

    Attributes
    ----------
    boosters : list of Booster
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(self, model_file: Optional[Union[str, Path]]=None):
        """Initialize the CVBooster.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the CVBooster model file.
        """
        self.boosters: List[Booster] = []
        self.best_iteration = -1
        if model_file is not None:
            with open(model_file, 'r') as file:
                self._from_dict(json.load(file))

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load CVBooster from dict."""
        self.best_iteration = models['best_iteration']
        self.boosters = []
        for model_str in models['boosters']:
            self.boosters.append(Booster(model_str=model_str))

    def _to_dict(self, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        """Serialize CVBooster to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(booster.model_to_string(num_iteration=num_iteration, start_iteration=start_iteration, importance_type=importance_type))
        return {'boosters': models_str, 'best_iteration': self.best_iteration}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        """Redirect methods call of CVBooster."""

        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def model_from_string(self, model_str: str) -> 'CVBooster':
        """Load CVBooster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : CVBooster
            Loaded CVBooster object.
        """
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(self, num_iteration: Optional[int]=None, start_iteration: int=0, importance_type: str='split') -> str:
        """Save CVBooster to JSON string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            JSON string representation of CVBooster.
        """
        return json.dumps(self._to_dict(num_iteration, start_iteration, importance_type))

    def save_model(self, filename: Union[str, Path], num_iteration: Optional[int]=None, start_iteration: int=0, importance_type: str='split') -> 'CVBooster':
        """Save CVBooster to a file as JSON text.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save CVBooster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : CVBooster
            Returns self.
        """
        with open(filename, 'w') as file:
            json.dump(self._to_dict(num_iteration, start_iteration, importance_type), file)
        return self