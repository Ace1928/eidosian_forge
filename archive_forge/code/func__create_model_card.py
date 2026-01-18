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
def _create_model_card(model, repo_dir: Path, plot_model: bool=True, metadata: Optional[dict]=None):
    """
    Creates a model card for the repository.

    Do not overwrite an existing README.md file.
    """
    readme_path = repo_dir / 'README.md'
    if readme_path.exists():
        return
    hyperparameters = _create_hyperparameter_table(model)
    if plot_model and is_graphviz_available() and is_pydot_available():
        _plot_network(model, repo_dir)
    if metadata is None:
        metadata = {}
    metadata['library_name'] = 'keras'
    model_card: str = '---\n'
    model_card += yaml_dump(metadata, default_flow_style=False)
    model_card += '---\n'
    model_card += '\n## Model description\n\nMore information needed\n'
    model_card += '\n## Intended uses & limitations\n\nMore information needed\n'
    model_card += '\n## Training and evaluation data\n\nMore information needed\n'
    if hyperparameters is not None:
        model_card += '\n## Training procedure\n'
        model_card += '\n### Training hyperparameters\n'
        model_card += '\nThe following hyperparameters were used during training:\n\n'
        model_card += hyperparameters
        model_card += '\n'
    if plot_model and os.path.exists(f'{repo_dir}/model.png'):
        model_card += '\n ## Model Plot\n'
        model_card += '\n<details>'
        model_card += '\n<summary>View Model Plot</summary>\n'
        path_to_plot = './model.png'
        model_card += f'\n![Model Image]({path_to_plot})\n'
        model_card += '\n</details>'
    readme_path.write_text(model_card)