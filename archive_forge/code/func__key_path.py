import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def _key_path(config_item: wandb_internal_pb2.ConfigItem) -> Sequence[str]:
    """Returns the key path referenced by the config item."""
    if config_item.nested_key:
        return config_item.nested_key
    elif config_item.key:
        return [config_item.key]
    else:
        raise AssertionError('Invalid ConfigItem: either key or nested_key must be set')