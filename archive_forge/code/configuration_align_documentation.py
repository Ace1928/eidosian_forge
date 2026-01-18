import os
from typing import TYPE_CHECKING, List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging

        Instantiate a [`AlignConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`AlignConfig`]: An instance of a configuration object
        