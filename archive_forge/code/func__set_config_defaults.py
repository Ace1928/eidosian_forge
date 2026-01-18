import warnings
from typing import Any, Dict, Optional, Union
from transformers import PretrainedConfig
def _set_config_defaults(self, config: Dict[str, Any], config_defaults: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in config_defaults.items():
        if k not in config:
            config[k] = v
    return config