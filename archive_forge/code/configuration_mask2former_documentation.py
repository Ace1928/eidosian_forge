from typing import Dict, List, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
Instantiate a [`Mask2FormerConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.

        Returns:
            [`Mask2FormerConfig`]: An instance of a configuration object
        