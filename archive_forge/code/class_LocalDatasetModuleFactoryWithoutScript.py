import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import fsspec
import requests
import yaml
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
from .splits import Split
from .utils import _datasets_server
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
from .utils.hub import hf_hub_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports
from .utils.version import Version
class LocalDatasetModuleFactoryWithoutScript(_DatasetModuleFactory):
    """Get the module of a dataset loaded from the user's data files. The dataset builder module to use is inferred
    from the data files extensions."""

    def __init__(self, path: str, data_dir: Optional[str]=None, data_files: Optional[Union[str, List, Dict]]=None, download_mode: Optional[Union[DownloadMode, str]]=None):
        if data_dir and os.path.isabs(data_dir):
            raise ValueError(f"`data_dir` must be relative to a dataset directory's root: {path}")
        self.path = Path(path).as_posix()
        self.name = Path(path).stem
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_mode = download_mode

    def get_module(self) -> DatasetModule:
        readme_path = os.path.join(self.path, config.REPOCARD_FILENAME)
        standalone_yaml_path = os.path.join(self.path, config.REPOYAML_FILENAME)
        dataset_card_data = DatasetCard.load(readme_path).data if os.path.isfile(readme_path) else DatasetCardData()
        if os.path.exists(standalone_yaml_path):
            with open(standalone_yaml_path, 'r', encoding='utf-8') as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        base_path = Path(self.path, self.data_dir or '').expanduser().resolve().as_posix()
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and (not self.data_dir) and ('data_files' in next(iter(metadata_configs.values()))):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))['data_files'])
        else:
            patterns = get_data_patterns(base_path)
        data_files = DataFilesDict.from_patterns(patterns, base_path=base_path, allowed_extensions=ALL_ALLOWED_EXTENSIONS)
        module_name, default_builder_kwargs = infer_module_for_data_files(data_files=data_files, path=self.path)
        data_files = data_files.filter_extensions(_MODULE_TO_EXTENSIONS[module_name])
        supports_metadata = module_name in _MODULE_SUPPORTS_METADATA
        if self.data_files is None and supports_metadata:
            try:
                metadata_patterns = get_metadata_patterns(base_path)
            except FileNotFoundError:
                metadata_patterns = None
            if metadata_patterns is not None:
                metadata_data_files_list = DataFilesList.from_patterns(metadata_patterns, base_path=base_path)
                if metadata_data_files_list:
                    data_files = DataFilesDict({split: data_files_list + metadata_data_files_list for split, data_files_list in data_files.items()})
        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(module_path, metadata_configs, base_path=base_path, supports_metadata=supports_metadata, default_builder_kwargs=default_builder_kwargs)
        else:
            builder_configs: List[BuilderConfig] = [import_main_class(module_path).BUILDER_CONFIG_CLASS(data_files=data_files, **default_builder_kwargs)]
            default_config_name = None
        builder_kwargs = {'base_path': self.path, 'dataset_name': camelcase_to_snakecase(Path(self.path).name)}
        if self.data_dir:
            builder_kwargs['data_files'] = data_files
        if os.path.isfile(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME), encoding='utf-8') as f:
                legacy_dataset_infos = DatasetInfosDict({config_name: DatasetInfo.from_dict(dataset_info_dict) for config_name, dataset_info_dict in json.load(f).items()})
                if len(legacy_dataset_infos) == 1:
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos['default'] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))
        hash = Hasher.hash({'dataset_infos': dataset_infos, 'builder_configs': builder_configs})
        return DatasetModule(module_path, hash, builder_kwargs, dataset_infos=dataset_infos, builder_configs_parameters=BuilderConfigsParameters(metadata_configs=metadata_configs, builder_configs=builder_configs, default_config_name=default_config_name))