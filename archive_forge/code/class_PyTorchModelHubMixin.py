import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Union, get_args
from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .utils import (
from .utils._deprecation import _deprecate_arguments
class PyTorchModelHubMixin(ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download capabilities to PyTorch models. The model
    is set in evaluation mode by default using `model.eval()` (dropout modules are deactivated). To train the model,
    you should first set it back in training mode with `model.train()`.

    Example:

    ```python
    >>> from dataclasses import dataclass
    >>> import torch
    >>> import torch.nn as nn
    >>> from huggingface_hub import PyTorchModelHubMixin

    >>> @dataclass
    ... class Config:
    ...     hidden_size: int = 512
    ...     vocab_size: int = 30000
    ...     output_size: int = 4

    >>> class MyModel(nn.Module, PyTorchModelHubMixin):
    ...     def __init__(self, config: Config):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(config.hidden_size, config.vocab_size))
    ...         self.linear = nn.Linear(config.output_size, config.vocab_size)

    ...     def forward(self, x):
    ...         return self.linear(x + self.param)
    >>> model = MyModel()

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    ```
    """

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, 'module') else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def _from_pretrained(cls, *, model_id: str, revision: Optional[str], cache_dir: Optional[Union[str, Path]], force_download: bool, proxies: Optional[Dict], resume_download: bool, local_files_only: bool, token: Union[str, bool, None], map_location: str='cpu', strict: bool=False, **model_kwargs):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print('Loading weights from local directory')
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(repo_id=model_id, filename=SAFETENSORS_SINGLE_FILE, revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, token=token, local_files_only=local_files_only)
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(repo_id=model_id, filename=PYTORCH_WEIGHTS_NAME, revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, token=token, local_files_only=local_files_only)
                return cls._load_as_pickle(model, model_file, map_location, strict)

    @classmethod
    def _load_as_pickle(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        load_model_as_safetensor(model, model_file, strict=strict)
        if map_location != 'cpu':
            logger.warning("Loading model weights on other devices than 'cpu' is not supported natively. This means that the model is loaded on 'cpu' first and then copied to the device. This leads to a slower loading time. Support for loading directly on other devices is planned to be added in future releases. See https://github.com/huggingface/huggingface_hub/pull/2086 for more details.")
            model.to(map_location)
        return model