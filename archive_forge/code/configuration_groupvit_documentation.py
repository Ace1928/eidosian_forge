import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

        Instantiate a [`GroupViTConfig`] (or a derived class) from groupvit text model configuration and groupvit
        vision model configuration.

        Returns:
            [`GroupViTConfig`]: An instance of a configuration object
        