from typing import Any, Dict, Optional
import ray.rllib.algorithms.appo.appo as appo
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/alpha_star/', new='rllib_contrib/alpha_star/', help=ALGO_DEPRECATION_WARNING, error=True)
class AlphaStar(appo.APPO):

    @classmethod
    @override(appo.APPO)
    def get_default_config(cls) -> AlphaStarConfig:
        return AlphaStarConfig()