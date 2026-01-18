from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
class ImpalaLearner(Learner):

    @override(Learner)
    def build(self) -> None:
        super().build()
        self.entropy_coeff_schedulers_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(lambda module_id: Scheduler(fixed_value_or_schedule=self.hps.get_hps_for_module(module_id).entropy_coeff, framework=self.framework, device=self._device))

    @override(Learner)
    def remove_module(self, module_id: str):
        super().remove_module(module_id)
        self.entropy_coeff_schedulers_per_module.pop(module_id)

    @override(Learner)
    def additional_update_for_module(self, *, module_id: ModuleID, hps: ImpalaLearnerHyperparameters, timestep: int) -> Dict[str, Any]:
        results = super().additional_update_for_module(module_id=module_id, hps=hps, timestep=timestep)
        new_entropy_coeff = self.entropy_coeff_schedulers_per_module[module_id].update(timestep=timestep)
        results.update({LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY: new_entropy_coeff})
        return results