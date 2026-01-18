from __future__ import print_function
import os
from copy import copy
from enum import Enum
from .. import CatBoostError, CatBoost
from .evaluation_result import EvaluationResults, MetricEvaluationResult
from ._fold_models_handler import FoldModelsHandler
from ._readers import _SimpleStreamingFileReader
from ._splitter import _Splitter
from .execution_case import ExecutionCase
from .factor_utils import LabelMode, FactorUtils
class EvalType(Enum):
    """
        Type of feature evaluation:
            All: All factors presented
            SeqRem:  Each factor while other presented
            SeqAdd:  Each factor while other removed
            SeqAddAndAll:  SeqAdd + All
    """
    All = 'All'
    SeqRem = 'SeqRem'
    SeqAdd = 'SeqAdd'
    SeqAddAndAll = 'SeqAddAndAll'