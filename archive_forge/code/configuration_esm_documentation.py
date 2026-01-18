from dataclasses import asdict, dataclass
from typing import Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging

        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        