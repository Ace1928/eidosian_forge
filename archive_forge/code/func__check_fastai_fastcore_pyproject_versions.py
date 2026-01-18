import json
import os
from pathlib import Path
from pickle import DEFAULT_PROTOCOL, PicklingError
from typing import Any, Dict, List, Optional, Union
from packaging import version
from huggingface_hub import snapshot_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import (
from .utils import logging, validate_hf_hub_args
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility...
def _check_fastai_fastcore_pyproject_versions(storage_folder: str, fastai_min_version: str='2.4', fastcore_min_version: str='1.3.27'):
    """
    Checks that the `pyproject.toml` file in the directory `storage_folder` has fastai and fastcore versions
    that are compatible with `from_pretrained_fastai` and `push_to_hub_fastai`. If `pyproject.toml` does not exist
    or does not contain versions for fastai and fastcore, then it logs a warning.

    Args:
        storage_folder (`str`):
            Folder to look for the `pyproject.toml` file.
        fastai_min_version (`str`, *optional*):
            The minimum fastai version supported.
        fastcore_min_version (`str`, *optional*):
            The minimum fastcore version supported.

    <Tip>
    Raises the following errors:

        - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
          if the `toml` module is not installed.
        - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
          if the `pyproject.toml` indicates a lower than minimum supported version of fastai or fastcore.

    </Tip>
    """
    try:
        import toml
    except ModuleNotFoundError:
        raise ImportError('`push_to_hub_fastai` and `from_pretrained_fastai` require the toml module. Install it with `pip install toml`.')
    if not os.path.isfile(f'{storage_folder}/pyproject.toml'):
        logger.warning('There is no `pyproject.toml` in the repository that contains the fastai `Learner`. The `pyproject.toml` would allow us to verify that your fastai and fastcore versions are compatible with those of the model you want to load.')
        return
    pyproject_toml = toml.load(f'{storage_folder}/pyproject.toml')
    if 'build-system' not in pyproject_toml.keys():
        logger.warning('There is no `build-system` section in the pyproject.toml of the repository that contains the fastai `Learner`. The `build-system` would allow us to verify that your fastai and fastcore versions are compatible with those of the model you want to load.')
        return
    build_system_toml = pyproject_toml['build-system']
    if 'requires' not in build_system_toml.keys():
        logger.warning('There is no `requires` section in the pyproject.toml of the repository that contains the fastai `Learner`. The `requires` would allow us to verify that your fastai and fastcore versions are compatible with those of the model you want to load.')
        return
    package_versions = build_system_toml['requires']
    fastai_packages = [pck for pck in package_versions if pck.startswith('fastai')]
    if len(fastai_packages) == 0:
        logger.warning('The repository does not have a fastai version specified in the `pyproject.toml`.')
    else:
        fastai_version = str(fastai_packages[0]).partition('=')[2]
        if fastai_version != '' and version.Version(fastai_version) < version.Version(fastai_min_version):
            raise ImportError(f'`from_pretrained_fastai` requires fastai>={fastai_min_version} version but the model to load uses {fastai_version} which is incompatible.')
    fastcore_packages = [pck for pck in package_versions if pck.startswith('fastcore')]
    if len(fastcore_packages) == 0:
        logger.warning('The repository does not have a fastcore version specified in the `pyproject.toml`.')
    else:
        fastcore_version = str(fastcore_packages[0]).partition('=')[2]
        if fastcore_version != '' and version.Version(fastcore_version) < version.Version(fastcore_min_version):
            raise ImportError(f'`from_pretrained_fastai` requires fastcore>={fastcore_min_version} version, but you are using fastcore version {fastcore_version} which is incompatible.')