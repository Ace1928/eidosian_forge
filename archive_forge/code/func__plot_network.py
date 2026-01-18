import collections.abc as collections
import json
import os
import warnings
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
def _plot_network(model, save_directory):
    tf.keras.utils.plot_model(model, to_file=f'{save_directory}/model.png', show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, layer_range=None)