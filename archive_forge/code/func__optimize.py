import contextlib
import math
import os
import warnings
from cupy._core import _optimize_config
from cupyx import profiler
def _optimize(optimize_config, target_func, suggest_func, default_best, ignore_error=()):
    assert isinstance(optimize_config, _optimize_config._OptimizationConfig)
    assert callable(target_func)
    assert callable(suggest_func)

    def objective(trial):
        args = suggest_func(trial)
        max_total_time = optimize_config.max_total_time_per_trial
        try:
            perf = profiler.benchmark(target_func, args, max_duration=max_total_time)
            return perf.gpu_times.mean()
        except Exception as e:
            if isinstance(e, ignore_error):
                return math.inf
            else:
                raise e
    study = optuna.create_study()
    study.enqueue_trial(default_best)
    study.optimize(objective, n_trials=optimize_config.max_trials, timeout=optimize_config.timeout)
    return study.best_trial