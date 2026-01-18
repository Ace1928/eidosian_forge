import pandas as pd
from fugue.workflow.workflow import FugueWorkflow
from pytest import raises
from tune import Space, MetricLogger
from tune.api.factory import (
from tune.concepts.dataset import TuneDataset
from tune.concepts.flow.judge import Monitor
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
from tune_optuna.optimizer import OptunaLocalOptimizer
def _nobjective(a: int) -> float:
    return 0.0