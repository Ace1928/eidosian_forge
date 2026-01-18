import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
class FakeRegNetParams(RegNetParams):
    """
    Used to instantiace a RegNet model from classy vision with the same depth as the 10B one but with super small
    parameters, so we can trace it in memory.
    """

    def get_expanded_params(self):
        return [(8, 2, 2, 8, 1.0), (8, 2, 7, 8, 1.0), (8, 2, 17, 8, 1.0), (8, 2, 1, 8, 1.0)]