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
def _check_periodic_pipeline(self, gen):
    """If enough time has passed, save a new optimized pipeline. Currently used in the per generation hook in the optimization loop.
        Parameters
        ----------
        gen: int
            Generation number

        Returns
        -------
        None
        """
    self._update_top_pipeline()
    if self.periodic_checkpoint_folder is not None:
        total_since_last_pipeline_save = (datetime.now() - self._last_pipeline_write).total_seconds()
        if total_since_last_pipeline_save > self._output_best_pipeline_period_seconds:
            self._last_pipeline_write = datetime.now()
            self._save_periodic_pipeline(gen)
    if self.early_stop is not None:
        if self._last_optimized_pareto_front_n_gens >= self.early_stop:
            raise StopIteration('The optimized pipeline was not improved after evaluating {} more generations. Will end the optimization process.\n'.format(self.early_stop))