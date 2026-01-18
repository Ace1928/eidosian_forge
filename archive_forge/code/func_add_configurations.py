import copy
import glob
import itertools
import os
import uuid
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.air._internal.usage import tag_searcher
from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI
def add_configurations(self, experiments: Union['Experiment', List['Experiment'], Dict[str, Dict]]):
    """Chains generator given experiment specifications.

        Arguments:
            experiments: Experiments to run.
        """
    from ray.tune.experiment import _convert_to_experiment_list
    experiment_list = _convert_to_experiment_list(experiments)
    for experiment in experiment_list:
        grid_vals = _count_spec_samples(experiment.spec, num_samples=1)
        lazy_eval = grid_vals > SERIALIZATION_THRESHOLD
        if lazy_eval:
            warnings.warn(f'The number of pre-generated samples ({grid_vals}) exceeds the serialization threshold ({int(SERIALIZATION_THRESHOLD)}). Resume ability is disabled. To fix this, reduce the number of dimensions/size of the provided grid search.')
        previous_samples = self._total_samples
        points_to_evaluate = copy.deepcopy(self._points_to_evaluate)
        self._total_samples += _count_variants(experiment.spec, points_to_evaluate)
        iterator = _TrialIterator(uuid_prefix=self._uuid_prefix, num_samples=experiment.spec.get('num_samples', 1), unresolved_spec=experiment.spec, constant_grid_search=self._constant_grid_search, points_to_evaluate=points_to_evaluate, lazy_eval=lazy_eval, start=previous_samples, random_state=self._random_state)
        self._iterators.append(iterator)
        self._trial_generator = itertools.chain(self._trial_generator, iterator)