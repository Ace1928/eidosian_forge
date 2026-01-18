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
def _preprocess_individuals(self, individuals):
    """Preprocess DEAP individuals before pipeline evaluation.

        Parameters
        ----------
        individuals: a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

        Returns
        -------
        operator_counts: dictionary
            a dictionary of operator counts in individuals for evaluation
        eval_individuals_str: list
            a list of string of individuals for evaluation
        sklearn_pipeline_list: list
            a list of scikit-learn pipelines converted from DEAP individuals for evaluation
        stats_dicts: dictionary
            A dict where 'key' is the string representation of an individual and 'value' is a dict containing statistics about the individual
        """
    if not self.max_time_mins is None and (not self._pbar.disable) and (self._pbar.total <= self._pbar.n):
        self._pbar.total += self._lambda
    _, unique_individual_indices = np.unique([str(ind) for ind in individuals], return_index=True)
    unique_individuals = [ind for i, ind in enumerate(individuals) if i in unique_individual_indices]
    self._update_pbar(pbar_num=len(individuals) - len(unique_individuals))
    operator_counts = {}
    stats_dicts = {}
    eval_individuals_str = []
    sklearn_pipeline_list = []
    for individual in unique_individuals:
        individual_str = str(individual)
        if not len(individual):
            self.evaluated_individuals_[individual_str] = self._combine_individual_stats(5000.0, -float('inf'), individual.statistics)
            self._update_pbar(pbar_msg='Invalid pipeline encountered. Skipping its evaluation.')
            continue
        sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(individual, self._pset), self.operators)
        if sklearn_pipeline_str.count('PolynomialFeatures') > 1:
            self.evaluated_individuals_[individual_str] = self._combine_individual_stats(5000.0, -float('inf'), individual.statistics)
            self._update_pbar(pbar_msg='Invalid pipeline encountered. Skipping its evaluation.')
        elif individual_str in self.evaluated_individuals_:
            self._update_pbar(pbar_msg='Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.')
        else:
            try:
                sklearn_pipeline = self._toolbox.compile(expr=individual)
                operator_count = self._operator_count(individual)
                operator_counts[individual_str] = max(1, operator_count)
                stats_dicts[individual_str] = individual.statistics
            except Exception:
                self.evaluated_individuals_[individual_str] = self._combine_individual_stats(5000.0, -float('inf'), individual.statistics)
                self._update_pbar()
                continue
            eval_individuals_str.append(individual_str)
            sklearn_pipeline_list.append(sklearn_pipeline)
    return (operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts)