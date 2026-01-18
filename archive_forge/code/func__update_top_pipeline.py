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
def _update_top_pipeline(self):
    """Helper function to update the _optimized_pipeline field."""
    if self._pareto_front:
        self._optimized_pipeline_score = -float('inf')
        for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
            if pipeline_scores.wvalues[1] > self._optimized_pipeline_score:
                self._optimized_pipeline = pipeline
                self._optimized_pipeline_score = pipeline_scores.wvalues[1]
        if not self._optimized_pipeline:
            eval_ind_list = list(self.evaluated_individuals_.keys())
            for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                if np.isinf(pipeline_scores.wvalues[1]):
                    sklearn_pipeline = self._toolbox.compile(expr=pipeline)
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(sklearn_pipeline, self.pretest_X, self.pretest_y, cv=self.cv, scoring=self.scoring_function, verbose=0, error_score='raise')
                    break
            raise RuntimeError('There was an error in the TPOT optimization process. This could be because the data was not formatted properly, because a timeout was reached or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/')
        else:
            pareto_front_wvalues = [pipeline_scores.wvalues[1] for pipeline_scores in self._pareto_front.keys]
            if not self._last_optimized_pareto_front:
                self._last_optimized_pareto_front = pareto_front_wvalues
            elif self._last_optimized_pareto_front == pareto_front_wvalues:
                self._last_optimized_pareto_front_n_gens += 1
            else:
                self._last_optimized_pareto_front = pareto_front_wvalues
                self._last_optimized_pareto_front_n_gens = 0
    else:
        raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')