from collections import OrderedDict, MutableMapping
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.wrappers.wrapper import EnvWrapper
def _update_name(self, name: str) -> str:
    return self.name