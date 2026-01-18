from collections import OrderedDict, MutableMapping
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.wrappers.wrapper import EnvWrapper
def _unwrap_action(self, act: OrderedDict) -> OrderedDict:
    for key, hdl in act:
        if '.' in key:
            act['key'] = flatten(act['key'])
    return act