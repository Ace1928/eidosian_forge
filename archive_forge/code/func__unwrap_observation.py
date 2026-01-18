from collections import OrderedDict, MutableMapping
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.wrappers.wrapper import EnvWrapper
def _unwrap_observation(self, obs: OrderedDict) -> OrderedDict:
    for key, hdl in obs:
        if '.' in key:
            obs['key'] = flatten(obs['key'])
    return obs