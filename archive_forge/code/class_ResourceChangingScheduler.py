from copy import deepcopy
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
import pickle
import warnings
from ray.air.execution.resources.request import _sum_bundles
from ray.util.annotations import PublicAPI
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.execution.placement_groups import PlacementGroupFactory
@PublicAPI(stability='beta')
class ResourceChangingScheduler(TrialScheduler):
    """A utility scheduler to dynamically change resources of live trials.

    .. versionadded:: 1.5.0

    .. note::
        Experimental. API may change in future releases.

    The ResourceChangingScheduler works by wrapping around any other
    scheduler and adjusting the resource requirements of live trials
    in response to the decisions of the wrapped scheduler
    through a user-specified ``resources_allocation_function``.

    An example of such a function can be found in
    :doc:`/tune/examples/includes/xgboost_dynamic_resources_example`.

    If the functional API is used, the current trial resources can be obtained
    by calling `tune.get_trial_resources()` inside the training function.
    The function should be able to
    :ref:`load and save checkpoints <tune-function-trainable-checkpointing>`
    (the latter preferably every iteration).

    If the Trainable (class) API is used, you can obtain the current trial
    resources through the ``Trainable.trial_resources`` property.

    Cannot be used if ``reuse_actors`` is True in ``tune.TuneConfig()``. A ValueError
    will be raised in that case.

    Args:
        base_scheduler: The scheduler to provide decisions
            about trials. If None, a default FIFOScheduler will be used.
        resources_allocation_function: The callable used to change
            live trial resource requiements during tuning. This callable
            will be called on each trial as it finishes one step of training.
            The callable must take four arguments: ``TrialRunner``, current
            ``Trial``, current result :class:`dict` and the
            ``ResourceChangingScheduler`` calling it. The callable must
            return a ``PlacementGroupFactory``
            or None (signifying no need for an update). If
            ``resources_allocation_function`` is None, no resource
            requirements will be changed at any time.
            By default, :class:`DistributeResources` will be used,
            distributing available CPUs and GPUs over all running trials
            in a robust way, without any prioritization.

    Warning:
        If the ``resources_allocation_function`` sets trial resource
        requirements to values bigger than possible, the trial will
        not run. Ensure that your callable accounts for that possibility
        by setting upper limits. Consult :class:`DistributeResources`
        to see how that may be done.

    Example:
        .. code-block:: python

            base_scheduler = ASHAScheduler(max_t=16)
            def my_resources_allocation_function(
                tune_controller: "TuneController",
                trial: Trial,
                result: Dict[str, Any],
                scheduler: "ResourceChangingScheduler"
            ) -> Optional[Union[PlacementGroupFactory, Resource]]:
                # logic here
                # usage of PlacementGroupFactory is strongly preferred
                return PlacementGroupFactory(...)
            scheduler = ResourceChangingScheduler(
                            base_scheduler,
                            my_resources_allocation_function
                        )

        See :doc:`/tune/examples/includes/xgboost_dynamic_resources_example` for a
        more detailed example.
    """

    def __init__(self, base_scheduler: Optional[TrialScheduler]=None, resources_allocation_function: Optional[Callable[['TuneController', Trial, Dict[str, Any], 'ResourceChangingScheduler'], Optional[PlacementGroupFactory]]]=_DistributeResourcesDefault) -> None:
        super().__init__()
        if resources_allocation_function is None:
            warnings.warn('`resources_allocation_function` is None. No resource requirements will be changed at any time. Pass a correctly defined function to enable functionality.')
        self._resources_allocation_function = resources_allocation_function
        self._base_scheduler = base_scheduler or FIFOScheduler()
        self._base_trial_resources: Optional[PlacementGroupFactory] = None
        self._trials_to_reallocate: Dict[Trial, Optional[Union[dict, PlacementGroupFactory]]] = {}
        self._reallocated_trial_ids: Set[str] = set()
        self._metric = None
        self._mode = None

    @property
    def metric(self):
        return self._base_scheduler._metric

    @property
    def base_trial_resources(self) -> Optional[PlacementGroupFactory]:
        return self._base_trial_resources

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], **spec) -> bool:
        self._metric = metric
        self._mode = mode
        return self._base_scheduler.set_search_properties(metric, mode, **spec)

    def on_trial_add(self, tune_controller: 'TuneController', trial: Trial, **kwargs):
        if self._base_trial_resources is None:
            self._base_trial_resources = trial.placement_group_factory
        elif trial.trial_id not in self._reallocated_trial_ids:
            trial_resources = trial.placement_group_factory
            if trial_resources != self._base_trial_resources:
                raise RuntimeError(f"ResourceChangingScheduler doesn't support trials with varying base resources. First trial had {self._base_trial_resources}, trial {trial} has {trial_resources}.")
        return self._base_scheduler.on_trial_add(tune_controller, trial, **kwargs)

    def on_trial_error(self, tune_controller: 'TuneController', trial: Trial, **kwargs):
        return self._base_scheduler.on_trial_error(tune_controller, trial, **kwargs)

    def on_trial_result(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> str:
        base_scheduler_decision = self._base_scheduler.on_trial_result(tune_controller, trial, result)
        if base_scheduler_decision == TrialScheduler.CONTINUE:
            new_resources = self.reallocate_trial_resources_if_needed(tune_controller, trial, result)
            if new_resources:
                self._trials_to_reallocate[trial] = new_resources
                return TrialScheduler.PAUSE
        return base_scheduler_decision

    def on_trial_complete(self, tune_controller: 'TuneController', trial: Trial, result: Dict, **kwargs):
        return self._base_scheduler.on_trial_complete(tune_controller, trial, result, **kwargs)

    def on_trial_remove(self, tune_controller: 'TuneController', trial: Trial, **kwargs):
        return self._base_scheduler.on_trial_remove(tune_controller, trial, **kwargs)

    def choose_trial_to_run(self, tune_controller: 'TuneController', **kwargs) -> Optional[Trial]:
        if getattr(tune_controller, '_reuse_actors', False):
            raise ValueError('ResourceChangingScheduler cannot be used with `reuse_actors=True`. FIX THIS by setting `reuse_actors=False` in `tune.TuneConfig()`.')
        any_resources_changed = False
        new_trials_to_reallocate = {}
        for trial, new_resources in self._trials_to_reallocate.items():
            if trial.status == Trial.RUNNING:
                new_trials_to_reallocate[trial] = new_resources
                logger.debug(f'{trial} is still running, skipping for now')
                continue
            any_resources_changed = any_resources_changed or self.set_trial_resources(trial, new_resources)
        self._trials_to_reallocate = new_trials_to_reallocate
        trial = self._base_scheduler.choose_trial_to_run(tune_controller, **kwargs)
        return trial

    def debug_string(self) -> str:
        return f'(ResourceChangingScheduler) {self._base_scheduler.debug_string()}'

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, 'rb') as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

    def set_trial_resources(self, trial: Trial, new_resources: Union[Dict, PlacementGroupFactory]) -> bool:
        """Returns True if new_resources were set."""
        if new_resources:
            logger.info(f'Setting trial {trial} resource to {new_resources} with {new_resources._bundles}')
            trial.placement_group_factory = None
            trial.update_resources(new_resources)
            self._reallocated_trial_ids.add(trial.trial_id)
            return True
        return False

    def _are_resources_the_same(self, trial: Trial, new_resources) -> bool:
        """Returns True if trial's resources are value equal to new_resources.

        Only checks for PlacementGroupFactories at this moment.
        """
        if isinstance(new_resources, PlacementGroupFactory) and trial.placement_group_factory == new_resources:
            logger.debug(f'{trial} PGF {trial.placement_group_factory.required_resources} and {new_resources.required_resources} are the same, skipping')
            return True
        else:
            return False

    def reallocate_trial_resources_if_needed(self, tune_controller: 'TuneController', trial: Trial, result: Dict) -> Optional[Union[dict, PlacementGroupFactory]]:
        """Calls user defined resources_allocation_function. If the returned
        resources are not none and not the same as currently present, returns
        them. Otherwise, returns None."""
        if self._resources_allocation_function is None:
            return None
        if not getattr(self._resources_allocation_function, 'metric', None):
            self._resources_allocation_function.metric = getattr(self._base_scheduler, '_metric', self._metric)
        if not getattr(self._resources_allocation_function, 'mode', None):
            self._resources_allocation_function.mode = getattr(self._base_scheduler, '_mode', self._mode)
        new_resources = self._resources_allocation_function(tune_controller, trial, result, self)
        if new_resources and (not self._are_resources_the_same(trial, new_resources)):
            return new_resources
        return None