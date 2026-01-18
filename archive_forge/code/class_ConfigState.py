import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
class ConfigState:
    """The configuration of a run."""

    def __init__(self, tree: Optional[Dict[str, Any]]=None) -> None:
        self._tree: Dict[str, Any] = tree or {}
        'A tree with string-valued nodes and JSON leaves.\n\n        Leaves are Python objects that are valid JSON values:\n\n        * Primitives like strings and numbers\n        * Dictionaries from strings to JSON objects\n        * Lists of JSON objects\n        '

    def non_internal_config(self) -> Dict[str, Any]:
        """Returns the config settings minus "_wandb"."""
        return {k: v for k, v in self._tree.items() if k != _WANDB_INTERNAL_KEY}

    def update_from_proto(self, config_record: wandb_internal_pb2.ConfigRecord) -> None:
        """Applies update and remove commands."""
        for config_item in config_record.update:
            self._update_at_path(_key_path(config_item), json.loads(config_item.value_json))
        for config_item in config_record.remove:
            self._delete_at_path(_key_path(config_item))

    def merge_resumed_config(self, old_config_tree: Dict[str, Any]) -> None:
        """Merges the config from a run that's being resumed."""
        self._add_unset_keys_from_subtree(old_config_tree, [])
        self._add_unset_keys_from_subtree(old_config_tree, [_WANDB_INTERNAL_KEY, 'visualize'])
        self._add_unset_keys_from_subtree(old_config_tree, [_WANDB_INTERNAL_KEY, 'viz'])

    def _add_unset_keys_from_subtree(self, old_config_tree: Dict[str, Any], path: Sequence[str]) -> None:
        """Uses the given subtree for keys that aren't already set."""
        old_subtree = _subtree(old_config_tree, path, create=False)
        if not old_subtree:
            return
        new_subtree = _subtree(self._tree, path, create=True)
        assert new_subtree is not None
        for key, value in old_subtree.items():
            if key not in new_subtree:
                new_subtree[key] = value

    def to_backend_dict(self, telemetry_record: telemetry.TelemetryRecord, framework: Optional[str], start_time_millis: int, metric_pbdicts: Sequence[Dict[int, Any]]) -> BackendConfigDict:
        """Returns a dictionary representation expected by the backend.

        The backend expects the configuration in a specific format, and the
        config is also used to store additional metadata about the run.

        Args:
            telemetry_record: Telemetry information to insert.
            framework: The detected framework used in the run (e.g. TensorFlow).
            start_time_millis: The run's start time in Unix milliseconds.
            metric_pbdicts: List of dict representations of metric protobuffers.
        """
        backend_dict = self._tree.copy()
        wandb_internal = backend_dict.setdefault(_WANDB_INTERNAL_KEY, {})
        py_version = telemetry_record.python_version
        if py_version:
            wandb_internal['python_version'] = py_version
        cli_version = telemetry_record.cli_version
        if cli_version:
            wandb_internal['cli_version'] = cli_version
        if framework:
            wandb_internal['framework'] = framework
        huggingface_version = telemetry_record.huggingface_version
        if huggingface_version:
            wandb_internal['huggingface_version'] = huggingface_version
        wandb_internal['is_jupyter_run'] = telemetry_record.env.jupyter
        wandb_internal['is_kaggle_kernel'] = telemetry_record.env.kaggle
        wandb_internal['start_time'] = start_time_millis
        wandb_internal['t'] = proto_util.proto_encode_to_dict(telemetry_record)
        if metric_pbdicts:
            wandb_internal['m'] = metric_pbdicts
        return BackendConfigDict({key: {'desc': None, 'value': value} for key, value in self._tree.items()})

    def _update_at_path(self, key_path: Sequence[str], value: Any) -> None:
        """Sets the value at the path in the config tree."""
        subtree = _subtree(self._tree, key_path[:-1], create=True)
        assert subtree is not None
        subtree[key_path[-1]] = value

    def _delete_at_path(self, key_path: Sequence[str]) -> None:
        """Removes the subtree at the path in the config tree."""
        subtree = _subtree(self._tree, key_path[:-1], create=False)
        if subtree:
            del subtree[key_path[-1]]