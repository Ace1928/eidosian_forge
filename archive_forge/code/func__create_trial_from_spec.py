import argparse
import json
from ray.tune.utils.serialization import TuneFunctionEncoder
from ray.train import CheckpointConfig
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.resources import json_to_resources
from ray.tune.utils.util import SafeFallbackEncoder
def _create_trial_from_spec(spec: dict, parser: argparse.ArgumentParser, **trial_kwargs):
    """Creates a Trial object from parsing the spec.

    Args:
        spec: A resolved experiment specification. Arguments should
            The args here should correspond to the command line flags
            in ray.tune.experiment.config_parser.
        parser: An argument parser object from
            make_parser.
        trial_kwargs: Extra keyword arguments used in instantiating the Trial.

    Returns:
        A trial object with corresponding parameters to the specification.
    """
    global _cached_pgf
    spec = spec.copy()
    resources = spec.pop('resources_per_trial', None)
    try:
        args, _ = parser.parse_known_args(_to_argv(spec))
    except SystemExit:
        raise TuneError('Error parsing args, see above message', spec)
    if resources:
        trial_kwargs['placement_group_factory'] = resources
    checkpoint_config = spec.get('checkpoint_config', CheckpointConfig())
    return Trial(trainable_name=spec['run'], config=spec.get('config', {}), stopping_criterion=spec.get('stop', {}), checkpoint_config=checkpoint_config, export_formats=spec.get('export_formats', []), restore_path=spec.get('restore'), trial_name_creator=spec.get('trial_name_creator'), trial_dirname_creator=spec.get('trial_dirname_creator'), log_to_file=spec.get('log_to_file'), max_failures=args.max_failures, storage=spec.get('storage'), **trial_kwargs)