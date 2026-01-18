from __future__ import print_function
import random
import inspect
import warnings
import sys
from functools import partial
from datetime import datetime
from multiprocessing import cpu_count
import os
import re
import errno
from tempfile import mkdtemp
from shutil import rmtree
import types
import numpy as np
from pandas import DataFrame
from scipy import sparse
import deap
from deap import base, creator, tools, gp
from copy import copy, deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_consistent_length, check_array
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import available_if
from joblib import Parallel, delayed, Memory
from update_checker import update_check
from ._version import __version__
from .operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from .export_utils import (
from .decorators import _pre_test
from .builtins import CombineDFs, StackingEstimator
from .config.classifier_light import classifier_config_dict_light
from .config.regressor_light import regressor_config_dict_light
from .config.classifier_mdr import tpot_mdr_classifier_config_dict
from .config.regressor_mdr import tpot_mdr_regressor_config_dict
from .config.regressor_sparse import regressor_config_sparse
from .config.classifier_sparse import classifier_config_sparse
from .config.classifier_nn import classifier_config_nn
from .config.classifier_cuml import classifier_config_cuml
from .config.regressor_cuml import regressor_config_cuml
from .metrics import SCORERS
from .gp_types import Output_Array
from .gp_deap import (
def _update_pbar(self, pbar_num=1, pbar_msg=None):
    """Update self._pbar and error message during pipeline evaluation.

        Parameters
        ----------
        pbar_num: int
            How many pipelines has been processed
        pbar_msg: None or string
            Error message

        Returns
        -------
        None
        """
    if not isinstance(self._pbar, type(None)):
        if self.verbosity > 2 and pbar_msg is not None:
            self._pbar.write(pbar_msg, file=self.log_file_)
        if not self._pbar.disable:
            self._pbar.update(pbar_num)