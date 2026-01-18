import copy
import logging
import os
from typing import Any, Dict, Tuple
from lxml import etree
import json
import numpy as np
from minerl.env._multiagent import _MultiAgentEnv
from minerl.env._singleagent import _SingleAgentEnv
from minerl.herobraine.env_specs.navigate_specs import Navigate
def _get_fake_obs(self) -> Dict[str, Any]:
    obs = {}
    info = {}
    for agent in self.task.agent_names:
        malmo_data = self._get_fake_malmo_data()
        pov = malmo_data['pov']
        del malmo_data['pov']
        pov = pov[::-1, :, :]
        _json_info = json.dumps(malmo_data)
        obs[agent], info[agent] = self._process_observation(agent, pov, _json_info)
    return (obs, info)