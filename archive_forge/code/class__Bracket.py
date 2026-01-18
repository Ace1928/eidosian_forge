import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
class _Bracket:
    """Logical object for tracking Hyperband bracket progress. Keeps track
    of proper parameters as designated by HyperBand.

    Also keeps track of progress to ensure good scheduling.
    """

    def __init__(self, time_attr: str, max_trials: int, init_t_attr: int, max_t_attr: int, eta: float, s: int, stop_last_trials: bool=True):
        self._live_trials = {}
        self._all_trials = []
        self._time_attr = time_attr
        self._n = self._n0 = max_trials
        self._r = self._r0 = init_t_attr
        self._max_t_attr = max_t_attr
        self._cumul_r = self._r0
        self._eta = eta
        self._halves = s
        self._total_work = self._calculate_total_work(self._n0, self._r0, s)
        self._completed_progress = 0
        self.stop_last_trials = stop_last_trials
        self.is_being_processed = False
        self.trials_to_unpause = set()

    def add_trial(self, trial: Trial):
        """Add trial to bracket assuming bracket is not filled.

        At a later iteration, a newly added trial will be given equal
        opportunity to catch up."""
        assert not self.filled(), 'Cannot add trial to filled bracket!'
        self._live_trials[trial] = None
        self._all_trials.append(trial)

    def cur_iter_done(self) -> bool:
        """Checks if all iterations have completed.

        TODO(rliaw): also check that `t.iterations == self._r`"""
        return all((self._get_result_time(result) >= self._cumul_r for result in self._live_trials.values()))

    def finished(self) -> bool:
        if not self.stop_last_trials:
            return False
        return self._halves == 0 and self.cur_iter_done()

    def current_trials(self) -> List[Trial]:
        return list(self._live_trials)

    def continue_trial(self, trial: Trial) -> bool:
        result = self._live_trials[trial]
        if not self.stop_last_trials and self._halves == 0:
            return True
        elif self._get_result_time(result) < self._cumul_r:
            logger.debug(f"Continuing trial {trial} as it hasn't reached the time threshold {self._cumul_r}, yet.")
            return True
        return False

    def filled(self) -> bool:
        """Checks if bracket is filled.

        Only let new trials be added at current level minimizing the need
        to backtrack and bookkeep previous medians."""
        return len(self._live_trials) == self._n

    def successive_halving(self, metric: str, metric_op: float) -> Tuple[List[Trial], List[Trial]]:
        if self._halves == 0 and (not self.stop_last_trials):
            return (self._live_trials, [])
        assert self._halves > 0
        self._halves -= 1
        self._n = int(np.ceil(self._n / self._eta))
        self._r *= self._eta
        self._r = int(min(self._r, self._max_t_attr))
        self._cumul_r = self._r
        sorted_trials = sorted(self._live_trials, key=lambda t: metric_op * self._live_trials[t][metric])
        good, bad = (sorted_trials[-self._n:], sorted_trials[:-self._n])
        return (good, bad)

    def update_trial_stats(self, trial: Trial, result: Dict):
        """Update result for trial. Called after trial has finished
        an iteration - will decrement iteration count.

        TODO(rliaw): The other alternative is to keep the trials
        in and make sure they're not set as pending later."""
        assert trial in self._live_trials
        assert self._get_result_time(result) >= 0
        observed_time = self._get_result_time(result)
        last_observed = self._get_result_time(self._live_trials[trial])
        delta = observed_time - last_observed
        if delta <= 0:
            logger.info('Restoring from a previous point in time. Previous={}; Now={}'.format(last_observed, observed_time))
        self._completed_progress += delta
        self._live_trials[trial] = result
        self.trials_to_unpause.discard(trial)

    def cleanup_trial(self, trial: Trial):
        """Clean up statistics tracking for terminated trials (either by force
        or otherwise).

        This may cause bad trials to continue for a long time, in the case
        where all the good trials finish early and there are only bad trials
        left in a bracket with a large max-iteration."""
        self._live_trials.pop(trial, None)

    def cleanup_full(self, tune_controller: 'TuneController'):
        """Cleans up bracket after bracket is completely finished.

        Lets the last trial continue to run until termination condition
        kicks in."""
        for trial in self.current_trials():
            if trial.status == Trial.PAUSED:
                tune_controller.stop_trial(trial)

    def completion_percentage(self) -> float:
        """Returns a progress metric.

        This will not be always finish with 100 since dead trials
        are dropped."""
        if self.finished():
            return 1.0
        return min(self._completed_progress / self._total_work, 1.0)

    def _get_result_time(self, result: Dict) -> float:
        if result is None:
            return 0
        return result[self._time_attr]

    def _calculate_total_work(self, n: int, r: float, s: int):
        work = 0
        cumulative_r = r
        for _ in range(s + 1):
            work += int(n) * int(r)
            n /= self._eta
            n = int(np.ceil(n))
            r *= self._eta
            r = int(min(r, self._max_t_attr - cumulative_r))
        return work

    def __repr__(self) -> str:
        status = ', '.join(['Max Size (n)={}'.format(self._n), 'Milestone (r)={}'.format(self._cumul_r), 'completed={:.1%}'.format(self.completion_percentage())])
        counts = collections.Counter([t.status for t in self._all_trials])
        trial_statuses = ', '.join(sorted(('{}: {}'.format(k, v) for k, v in counts.items())))
        return 'Bracket({}): {{{}}} '.format(status, trial_statuses)