from collections import OrderedDict, MutableMapping
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.wrappers.wrapper import EnvWrapper
class Compat_v0(EnvWrapper):

    def _update_name(self, name: str) -> str:
        return self.name

    def _wrap_observation(self, obs: OrderedDict) -> OrderedDict:
        for key, hdl in obs:
            if '.' in key:
                obs['key'] = flatten(obs['key'])
        return obs

    def _wrap_action(self, act: OrderedDict) -> OrderedDict:
        for key, hdl in act:
            if '.' in key:
                act['key'] = flatten(act['key'])
        return act

    def _unwrap_observation(self, obs: OrderedDict) -> OrderedDict:
        for key, hdl in obs:
            if '.' in key:
                obs['key'] = flatten(obs['key'])
        return obs

    def _unwrap_action(self, act: OrderedDict) -> OrderedDict:
        for key, hdl in act:
            if '.' in key:
                act['key'] = flatten(act['key'])
        return act

    def __init__(self, env_to_wrap: EnvSpec, name: str):
        super().__init__(env_to_wrap)
        self.name = name
        pass