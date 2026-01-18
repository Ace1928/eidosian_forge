import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

        Instantiate a [`OwlViTConfig`] (or a derived class) from owlvit text model configuration and owlvit vision
        model configuration.

        Returns:
            [`OwlViTConfig`]: An instance of a configuration object
        