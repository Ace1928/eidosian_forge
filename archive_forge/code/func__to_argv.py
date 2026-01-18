import argparse
import json
from ray.tune.utils.serialization import TuneFunctionEncoder
from ray.train import CheckpointConfig
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.resources import json_to_resources
from ray.tune.utils.util import SafeFallbackEncoder
def _to_argv(config):
    """Converts configuration to a command line argument format."""
    argv = []
    for k, v in config.items():
        if '-' in k:
            raise ValueError("Use '_' instead of '-' in `{}`".format(k))
        if v is None:
            continue
        if not isinstance(v, bool) or v:
            argv.append('--{}'.format(k.replace('_', '-')))
        if isinstance(v, str):
            argv.append(v)
        elif isinstance(v, bool):
            pass
        elif callable(v):
            argv.append(json.dumps(v, cls=TuneFunctionEncoder))
        else:
            argv.append(json.dumps(v, cls=SafeFallbackEncoder))
    return argv