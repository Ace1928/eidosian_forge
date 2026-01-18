import os
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING

        Instantiate a [`BarkConfig`] (or a derived class) from bark sub-models configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        