from typing import Any
from hyperopt.early_stop import no_progress_loss
from tune import parse_noniterative_local_optimizer
from tune.noniterative.objective import NonIterativeObjectiveLocalOptimizer
from tune_test.local_optmizer import NonIterativeObjectiveLocalOptimizerTests
from tune_hyperopt import HyperoptLocalOptimizer
class HyperoptLocalOptimizerTests(NonIterativeObjectiveLocalOptimizerTests.Tests):

    def make_optimizer(self, **kwargs: Any) -> NonIterativeObjectiveLocalOptimizer:
        kwargs = {'seed': 0, 'kwargs_func': _add_conf, **kwargs}
        return HyperoptLocalOptimizer(**kwargs)