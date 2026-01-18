import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def _process_bracket(self, tune_controller: 'TuneController', bracket: '_Bracket') -> str:
    """This is called whenever a trial makes progress.

        When all live trials in the bracket have no more iterations left,
        Trials will be successively halved. If bracket is done, all
        non-running trials will be stopped and cleaned up,
        and during each halving phase, bad trials will be stopped while good
        trials will return to "PENDING".

        Note some implicit conditions here: In ``on_trial_result`` a trial is
        either continued (e.g. if it didn't reach the time threshold for the bracket)
        or this method (``_process_bracket``) is called. If there are other trials left
        that still haven't reached the threshold, the trial is PAUSED. This means
        that when the bracket is actually processed (``bracket.cur_iter_done``), there
        is at most one RUNNING trial (which is the trial that is currently processed)
        and the rest are either PAUSED (as explained above) or TERMINATED/ERRORED
        (if they finish separately).
        """
    action = TrialScheduler.PAUSE
    if bracket.cur_iter_done():
        if bracket.finished():
            bracket.cleanup_full(tune_controller)
            return TrialScheduler.STOP
        bracket.is_being_processed = True
        good, bad = bracket.successive_halving(self._metric, self._metric_op)
        logger.debug(f'Processing {len(good)} good and {len(bad)} bad trials in bracket {bracket}.\nGood: {good}\nBad: {bad}')
        self._num_stopped += len(bad)
        for t in bad:
            if t.status == Trial.PAUSED or t.is_saving:
                logger.debug(f'Stopping other trial {str(t)}')
                tune_controller.stop_trial(t)
            elif t.status == Trial.RUNNING:
                logger.debug(f'Stopping current trial {str(t)}')
                bracket.cleanup_trial(t)
                action = TrialScheduler.STOP
            else:
                raise TuneError(f'Trial with unexpected bad status encountered: {str(t)} is {t.status}')
        for t in good:
            if bracket.continue_trial(t):
                assert t.status not in (Trial.ERROR, Trial.TERMINATED), f'Good trial {t.trial_id} is in an invalid state: {t.status}\nExpected trial to be either PAUSED, PENDING, or RUNNING.\nIf you encounter this, please file an issue on the Ray Github.'
                if t.status == Trial.PAUSED or t.is_saving:
                    logger.debug(f'Unpausing trial {str(t)}')
                    self._unpause_trial(tune_controller, t)
                    bracket.trials_to_unpause.add(t)
                elif t.status == Trial.RUNNING:
                    logger.debug(f'Continuing current trial {str(t)}')
                    action = TrialScheduler.CONTINUE
            elif bracket.finished() and bracket.stop_last_trials:
                if t.status == Trial.PAUSED or t.is_saving:
                    logger.debug(f'Bracket finished. Stopping other trial {str(t)}')
                    tune_controller.stop_trial(t)
                elif t.status == Trial.RUNNING:
                    logger.debug(f'Bracket finished. Stopping current trial {str(t)}')
                    bracket.cleanup_trial(t)
                    action = TrialScheduler.STOP
    return action