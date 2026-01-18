import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
class DatasetBuilder:
    """Abstract base class for all datasets.

    `DatasetBuilder` has 3 key methods:

        - [`DatasetBuilder.info`]: Documents the dataset, including feature
          names, types, shapes, version, splits, citation, etc.
        - [`DatasetBuilder.download_and_prepare`]: Downloads the source data
          and writes it to disk.
        - [`DatasetBuilder.as_dataset`]: Generates a [`Dataset`].

    Some `DatasetBuilder`s expose multiple variants of the
    dataset by defining a [`BuilderConfig`] subclass and accepting a
    config object (or name) on construction. Configurable datasets expose a
    pre-defined set of configurations in [`DatasetBuilder.builder_configs`].

    Args:
        cache_dir (`str`, *optional*):
            Directory to cache data. Defaults to `"~/.cache/huggingface/datasets"`.
        dataset_name (`str`, *optional*):
            Name of the dataset, if different from the builder name. Useful for packaged builders
            like csv, imagefolder, audiofolder, etc. to reflect the difference between datasets
            that use the same packaged builder.
        config_name (`str`, *optional*):
            Name of the dataset configuration.
            It affects the data generated on disk. Different configurations will have their own subdirectories and
            versions.
            If not provided, the default configuration is used (if it exists).

            <Added version="2.3.0">

            Parameter `name` was renamed to `config_name`.

            </Added>
        hash (`str`, *optional*):
            Hash specific to the dataset code. Used to update the caching directory when the
            dataset loading script code is updated (to avoid reusing old data).
            The typical caching directory (defined in `self._relative_data_dir`) is `name/version/hash/`.
        base_path (`str`, *optional*):
            Base path for relative paths that are used to download files.
            This can be a remote URL.
        features ([`Features`], *optional*):
            Features types to use with this dataset.
            It can be used to change the [`Features`] types of a dataset, for example.
        token (`str` or `bool`, *optional*):
            String or boolean to use as Bearer token for remote files on the
            Datasets Hub. If `True`, will get token from `"~/.huggingface"`.
        repo_id (`str`, *optional*):
            ID of the dataset repository.
            Used to distinguish builders with the same name but not coming from the same namespace, for example "squad"
            and "lhoestq/squad" repo IDs. In the latter, the builder name would be "lhoestq___squad".
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
            For builders like "csv" or "json" that need the user to specify data files. They can be either
            local or remote files. For convenience, you can use a `DataFilesDict`.
        data_dir (`str`, *optional*):
            Path to directory containing source data file(s).
            Use only if `data_files` is not passed, in which case it is equivalent to passing
            `os.path.join(data_dir, "**")` as `data_files`.
            For builders that require manual download, it must be the path to the local directory containing the
            manually downloaded data.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        writer_batch_size (`int`, *optional*):
            Batch size used by the ArrowWriter.
            It defines the number of samples that are kept in memory before writing them
            and also the length of the arrow chunks.
            None means that the ArrowWriter will use its default value.
        name (`str`): Configuration name for the dataset.

            <Deprecated version="2.3.0">

            Use `config_name` instead.

            </Deprecated>

        **config_kwargs (additional keyword arguments): Keyword arguments to be passed to the corresponding builder
            configuration class, set on the class attribute [`DatasetBuilder.BUILDER_CONFIG_CLASS`]. The builder
            configuration class is [`BuilderConfig`] or a subclass of it.
    """
    VERSION = None
    BUILDER_CONFIG_CLASS = BuilderConfig
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = None
    DEFAULT_WRITER_BATCH_SIZE = None

    def __init__(self, cache_dir: Optional[str]=None, dataset_name: Optional[str]=None, config_name: Optional[str]=None, hash: Optional[str]=None, base_path: Optional[str]=None, info: Optional[DatasetInfo]=None, features: Optional[Features]=None, token: Optional[Union[bool, str]]=None, use_auth_token='deprecated', repo_id: Optional[str]=None, data_files: Optional[Union[str, list, dict, DataFilesDict]]=None, data_dir: Optional[str]=None, storage_options: Optional[dict]=None, writer_batch_size: Optional[int]=None, name='deprecated', **config_kwargs):
        if use_auth_token != 'deprecated':
            warnings.warn(f"'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'token={use_auth_token}' instead.", FutureWarning)
            token = use_auth_token
        if name != 'deprecated':
            warnings.warn("Parameter 'name' was renamed to 'config_name' in version 2.3.0 and will be removed in 3.0.0.", category=FutureWarning)
            config_name = name
        self.name: str = camelcase_to_snakecase(self.__module__.split('.')[-1])
        self.hash: Optional[str] = hash
        self.base_path = base_path
        self.token = token
        self.use_auth_token = token
        self.repo_id = repo_id
        self.storage_options = storage_options or {}
        self.dataset_name = camelcase_to_snakecase(dataset_name) if dataset_name else self.name
        self._writer_batch_size = writer_batch_size or self.DEFAULT_WRITER_BATCH_SIZE
        if data_files is not None and (not isinstance(data_files, DataFilesDict)):
            data_files = DataFilesDict.from_patterns(sanitize_patterns(data_files), base_path=base_path, download_config=DownloadConfig(token=token, storage_options=self.storage_options))
        if 'features' in inspect.signature(self.BUILDER_CONFIG_CLASS.__init__).parameters and features is not None:
            config_kwargs['features'] = features
        if data_files is not None:
            config_kwargs['data_files'] = data_files
        if data_dir is not None:
            config_kwargs['data_dir'] = data_dir
        self.config, self.config_id = self._create_builder_config(config_name=config_name, custom_features=features, **config_kwargs)
        if info is None:
            info = self.get_exported_dataset_info()
            info.update(self._info())
        info.builder_name = self.name
        info.dataset_name = self.dataset_name
        info.config_name = self.config.name
        info.version = self.config.version
        self.info = info
        if features is not None:
            self.info.features = features
        self._cache_dir_root = str(cache_dir or config.HF_DATASETS_CACHE)
        self._cache_dir_root = self._cache_dir_root if is_remote_url(self._cache_dir_root) else os.path.expanduser(self._cache_dir_root)
        self._cache_downloaded_dir = posixpath.join(self._cache_dir_root, config.DOWNLOADED_DATASETS_DIR) if cache_dir else str(config.DOWNLOADED_DATASETS_PATH)
        self._cache_downloaded_dir = self._cache_downloaded_dir if is_remote_url(self._cache_downloaded_dir) else os.path.expanduser(self._cache_downloaded_dir)
        self._legacy_relative_data_dir = None
        self._cache_dir = self._build_cache_dir()
        if not is_remote_url(self._cache_dir_root):
            os.makedirs(self._cache_dir_root, exist_ok=True)
            lock_path = os.path.join(self._cache_dir_root, Path(self._cache_dir).as_posix().replace('/', '_') + '.lock')
            with FileLock(lock_path):
                if os.path.exists(self._cache_dir):
                    if len(os.listdir(self._cache_dir)) > 0:
                        if os.path.exists(os.path.join(self._cache_dir, config.DATASET_INFO_FILENAME)):
                            logger.info('Overwrite dataset info from restored data version if exists.')
                            self.info = DatasetInfo.from_directory(self._cache_dir)
                    else:
                        logger.warning(f'Old caching folder {self._cache_dir} for dataset {self.dataset_name} exists but no data were found. Removing it. ')
                        os.rmdir(self._cache_dir)
        self._output_dir = self._cache_dir
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem('file')
        self.dl_manager = None
        self._record_infos = False
        self._file_format = None
        extend_dataset_builder_for_streaming(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        extend_dataset_builder_for_streaming(self)

    @property
    def manual_download_instructions(self) -> Optional[str]:
        return None

    def _check_legacy_cache(self) -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{builder_name} from 2.13"""
        if self.__module__.startswith('datasets.') and (not is_remote_url(self._cache_dir_root)) and (self.config.name == 'default'):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES
            namespace = self.repo_id.split('/')[0] if self.repo_id and self.repo_id.count('/') > 0 else None
            config_name = self.repo_id.replace('/', '--') if self.repo_id is not None else self.dataset_name
            config_id = config_name + self.config_id[len(self.config.name):]
            hash = _PACKAGED_DATASETS_MODULES.get(self.name, 'missing')[1]
            legacy_relative_data_dir = posixpath.join(self.dataset_name if namespace is None else f'{namespace}___{self.dataset_name}', config_id, '0.0.0', hash)
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    def _check_legacy_cache2(self, dataset_module: 'DatasetModule') -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{dataset_name}/{config_name}-xxx from 2.14 and 2.15"""
        if self.__module__.startswith('datasets.') and (not is_remote_url(self._cache_dir_root)):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES
            from .utils._dill import Pickler

            def update_hash_with_config_parameters(hash: str, config_parameters: dict) -> str:
                """
                Used to update hash of packaged modules which is used for creating unique cache directories to reflect
                different config parameters which are passed in metadata from readme.
                """
                params_to_exclude = {'config_name', 'version', 'description'}
                params_to_add_to_hash = {param: value for param, value in sorted(config_parameters.items()) if param not in params_to_exclude}
                m = Hasher()
                m.update(hash)
                m.update(params_to_add_to_hash)
                return m.hexdigest()
            namespace = self.repo_id.split('/')[0] if self.repo_id and self.repo_id.count('/') > 0 else None
            with patch.object(Pickler, '_legacy_no_dict_keys_sorting', True):
                config_id = self.config.name + '-' + Hasher.hash({'data_files': self.config.data_files})
            hash = _PACKAGED_DATASETS_MODULES.get(self.name, 'missing')[1]
            if dataset_module.builder_configs_parameters.metadata_configs and self.config.name in dataset_module.builder_configs_parameters.metadata_configs:
                hash = update_hash_with_config_parameters(hash, dataset_module.builder_configs_parameters.metadata_configs[self.config.name])
            legacy_relative_data_dir = posixpath.join(self.dataset_name if namespace is None else f'{namespace}___{self.dataset_name}', config_id, '0.0.0', hash)
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    @classmethod
    def get_all_exported_dataset_infos(cls) -> DatasetInfosDict:
        """Empty dict if doesn't exist

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> ds_builder = load_dataset_builder('rotten_tomatoes')
        >>> ds_builder.get_all_exported_dataset_infos()
        {'default': DatasetInfo(description="Movie Review Dataset.
This is a dataset of containing 5,331 positive and 5,331 negative processed
sentences from Rotten Tomatoes movie reviews. This data was first used in Bo
Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for
sentiment categorization with respect to rating scales.'', Proceedings of the
ACL, 2005.
", citation='@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
', homepage='http://www.cs.cornell.edu/people/pabo/movie-review-data/', license='', features={'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}, post_processed=None, supervised_keys=SupervisedKeysData(input='', output=''), task_templates=[TextClassification(task='text-classification', text_column='text', label_column='label')], builder_name='rotten_tomatoes_movie_review', config_name='default', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=1074810, num_examples=8530, dataset_name='rotten_tomatoes_movie_review'), 'validation': SplitInfo(name='validation', num_bytes=134679, num_examples=1066, dataset_name='rotten_tomatoes_movie_review'), 'test': SplitInfo(name='test', num_bytes=135972, num_examples=1066, dataset_name='rotten_tomatoes_movie_review')}, download_checksums={'https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz': {'num_bytes': 487770, 'checksum': 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'}}, download_size=487770, post_processing_size=None, dataset_size=1345461, size_in_bytes=1833231)}
        ```
        """
        return DatasetInfosDict.from_directory(cls.get_imported_module_dir())

    def get_exported_dataset_info(self) -> DatasetInfo:
        """Empty `DatasetInfo` if doesn't exist

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> ds_builder = load_dataset_builder('rotten_tomatoes')
        >>> ds_builder.get_exported_dataset_info()
        DatasetInfo(description="Movie Review Dataset.
This is a dataset of containing 5,331 positive and 5,331 negative processed
sentences from Rotten Tomatoes movie reviews. This data was first used in Bo
Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for
sentiment categorization with respect to rating scales.'', Proceedings of the
ACL, 2005.
", citation='@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
', homepage='http://www.cs.cornell.edu/people/pabo/movie-review-data/', license='', features={'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}, post_processed=None, supervised_keys=SupervisedKeysData(input='', output=''), task_templates=[TextClassification(task='text-classification', text_column='text', label_column='label')], builder_name='rotten_tomatoes_movie_review', config_name='default', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=1074810, num_examples=8530, dataset_name='rotten_tomatoes_movie_review'), 'validation': SplitInfo(name='validation', num_bytes=134679, num_examples=1066, dataset_name='rotten_tomatoes_movie_review'), 'test': SplitInfo(name='test', num_bytes=135972, num_examples=1066, dataset_name='rotten_tomatoes_movie_review')}, download_checksums={'https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz': {'num_bytes': 487770, 'checksum': 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'}}, download_size=487770, post_processing_size=None, dataset_size=1345461, size_in_bytes=1833231)
        ```
        """
        return self.get_all_exported_dataset_infos().get(self.config.name, DatasetInfo())

    def _create_builder_config(self, config_name=None, custom_features=None, **config_kwargs) -> Tuple[BuilderConfig, str]:
        """Create and validate BuilderConfig object as well as a unique config id for this config.
        Raises ValueError if there are multiple builder configs and config_name and DEFAULT_CONFIG_NAME are None.
        config_kwargs override the defaults kwargs in config
        """
        builder_config = None
        if config_name is None and self.BUILDER_CONFIGS:
            if self.DEFAULT_CONFIG_NAME is not None:
                builder_config = self.builder_configs.get(self.DEFAULT_CONFIG_NAME)
                logger.info(f'No config specified, defaulting to: {self.dataset_name}/{builder_config.name}')
            elif len(self.BUILDER_CONFIGS) > 1:
                if not config_kwargs:
                    example_of_usage = f"load_dataset('{self.dataset_name}', '{self.BUILDER_CONFIGS[0].name}')"
                    raise ValueError(f'Config name is missing.\nPlease pick one among the available configs: {list(self.builder_configs.keys())}' + f'\nExample of usage:\n\t`{example_of_usage}`')
            else:
                builder_config = self.BUILDER_CONFIGS[0]
                logger.info(f'No config specified, defaulting to the single config: {self.dataset_name}/{builder_config.name}')
        if isinstance(config_name, str):
            builder_config = self.builder_configs.get(config_name)
            if builder_config is None and self.BUILDER_CONFIGS:
                raise ValueError(f"BuilderConfig '{config_name}' not found. Available: {list(self.builder_configs.keys())}")
        if not builder_config:
            if config_name is not None:
                config_kwargs['name'] = config_name
            elif self.DEFAULT_CONFIG_NAME and (not config_kwargs):
                config_kwargs['name'] = self.DEFAULT_CONFIG_NAME
            if 'version' not in config_kwargs and hasattr(self, 'VERSION') and self.VERSION:
                config_kwargs['version'] = self.VERSION
            builder_config = self.BUILDER_CONFIG_CLASS(**config_kwargs)
        else:
            builder_config = copy.deepcopy(builder_config) if config_kwargs else builder_config
            for key, value in config_kwargs.items():
                if value is not None:
                    if not hasattr(builder_config, key):
                        raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                    setattr(builder_config, key, value)
        if not builder_config.name:
            raise ValueError(f'BuilderConfig must have a name, got {builder_config.name}')
        builder_config._resolve_data_files(base_path=self.base_path, download_config=DownloadConfig(token=self.token, storage_options=self.storage_options))
        config_id = builder_config.create_config_id(config_kwargs, custom_features=custom_features)
        is_custom = config_id not in self.builder_configs and config_id != 'default'
        if is_custom:
            logger.info(f'Using custom data configuration {config_id}')
        else:
            if builder_config.name in self.builder_configs and builder_config != self.builder_configs[builder_config.name]:
                raise ValueError(f'Cannot name a custom BuilderConfig the same as an available BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}')
            if not builder_config.version:
                raise ValueError(f'BuilderConfig {builder_config.name} must have a version')
        return (builder_config, config_id)

    @classproperty
    @classmethod
    @memoize()
    def builder_configs(cls) -> Dict[str, BuilderConfig]:
        """Dictionary of pre-defined configurations for this builder class."""
        configs = {config.name: config for config in cls.BUILDER_CONFIGS}
        if len(configs) != len(cls.BUILDER_CONFIGS):
            names = [config.name for config in cls.BUILDER_CONFIGS]
            raise ValueError(f'Names in BUILDER_CONFIGS must not be duplicated. Got {names}')
        return configs

    @property
    def cache_dir(self):
        return self._cache_dir

    def _use_legacy_cache_dir_if_possible(self, dataset_module: 'DatasetModule'):
        self._legacy_relative_data_dir = self._check_legacy_cache2(dataset_module) or self._check_legacy_cache() or None
        self._cache_dir = self._build_cache_dir()
        self._output_dir = self._cache_dir

    def _relative_data_dir(self, with_version=True, with_hash=True) -> str:
        """Relative path of this dataset in cache_dir:
        Will be:
            self.dataset_name/self.config.version/self.hash/
        or if a repo_id with a namespace has been specified:
            self.namespace___self.dataset_name/self.config.version/self.hash/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
        if self._legacy_relative_data_dir is not None and with_version and with_hash:
            return self._legacy_relative_data_dir
        namespace = self.repo_id.split('/')[0] if self.repo_id and self.repo_id.count('/') > 0 else None
        builder_data_dir = self.dataset_name if namespace is None else f'{namespace}___{self.dataset_name}'
        builder_data_dir = posixpath.join(builder_data_dir, self.config_id)
        if with_version:
            builder_data_dir = posixpath.join(builder_data_dir, str(self.config.version))
        if with_hash and self.hash and isinstance(self.hash, str):
            builder_data_dir = posixpath.join(builder_data_dir, self.hash)
        return builder_data_dir

    def _build_cache_dir(self):
        """Return the data directory for the current version."""
        builder_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=False))
        version_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=True))

        def _other_versions_on_disk():
            """Returns previous versions on disk."""
            if not os.path.exists(builder_data_dir):
                return []
            version_dirnames = []
            for dir_name in os.listdir(builder_data_dir):
                try:
                    version_dirnames.append((utils.Version(dir_name), dir_name))
                except ValueError:
                    pass
            version_dirnames.sort(reverse=True)
            return version_dirnames
        if not is_remote_url(builder_data_dir):
            version_dirs = _other_versions_on_disk()
            if version_dirs:
                other_version = version_dirs[0][0]
                if other_version != self.config.version:
                    warn_msg = f'Found a different version {str(other_version)} of dataset {self.dataset_name} in cache_dir {self._cache_dir_root}. Using currently defined version {str(self.config.version)}.'
                    logger.warning(warn_msg)
        return version_data_dir

    @abc.abstractmethod
    def _info(self) -> DatasetInfo:
        """Construct the DatasetInfo object. See `DatasetInfo` for details.

        Warning: This function is only called once and the result is cached for all
        following .info() calls.

        Returns:
            info: (DatasetInfo) The dataset information
        """
        raise NotImplementedError

    @classmethod
    def get_imported_module_dir(cls):
        """Return the path of the module of this class or subclass."""
        return os.path.dirname(inspect.getfile(inspect.getmodule(cls)))

    def _rename(self, src: str, dst: str):
        rename(self._fs, src, dst)

    def download_and_prepare(self, output_dir: Optional[str]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, verification_mode: Optional[Union[VerificationMode, str]]=None, ignore_verifications='deprecated', try_from_hf_gcs: bool=True, dl_manager: Optional[DownloadManager]=None, base_path: Optional[str]=None, use_auth_token='deprecated', file_format: str='arrow', max_shard_size: Optional[Union[int, str]]=None, num_proc: Optional[int]=None, storage_options: Optional[dict]=None, **download_and_prepare_kwargs):
        """Downloads and prepares dataset for reading.

        Args:
            output_dir (`str`, *optional*):
                Output directory for the dataset.
                Default to this builder's `cache_dir`, which is inside `~/.cache/huggingface/datasets` by default.

                <Added version="2.5.0"/>
            download_config (`DownloadConfig`, *optional*):
                Specific download configuration parameters.
            download_mode ([`DownloadMode`] or `str`, *optional*):
                Select the download/generate mode, default to `REUSE_DATASET_IF_EXISTS`.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            try_from_hf_gcs (`bool`):
                If `True`, it will try to download the already prepared dataset from the HF Google cloud storage.
            dl_manager (`DownloadManager`, *optional*):
                Specific `DownloadManger` to use.
            base_path (`str`, *optional*):
                Base path for relative paths that are used to download files. This can be a remote url.
                If not specified, the value of the `base_path` attribute (`self.base_path`) will be used instead.
            use_auth_token (`Union[str, bool]`, *optional*):
                Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
                If True, or not specified, will get token from ~/.huggingface.

                <Deprecated version="2.7.1">

                Pass `use_auth_token` to `load_dataset_builder` instead.

                </Deprecated>
            file_format (`str`, *optional*):
                Format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
                If the format is "parquet", then image and audio data are embedded into the Parquet files instead of pointing to local files.

                <Added version="2.5.0"/>
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.

                <Added version="2.5.0"/>
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the caching file-system backend, if any.

                <Added version="2.5.0"/>
            **download_and_prepare_kwargs (additional keyword arguments): Keyword arguments.

        Example:

        Download and prepare the dataset as Arrow files that can be loaded as a Dataset using `builder.as_dataset()`:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare()
        ```

        Download and prepare the dataset as sharded Parquet files locally:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare("./output_dir", file_format="parquet")
        ```

        Download and prepare the dataset as sharded Parquet files in a cloud storage:

        ```py
        >>> from datasets import load_dataset_builder
        >>> storage_options = {"key": aws_access_key_id, "secret": aws_secret_access_key}
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare("s3://my-bucket/my_rotten_tomatoes", storage_options=storage_options, file_format="parquet")
        ```
        """
        if ignore_verifications != 'deprecated':
            verification_mode = VerificationMode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(f"'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.\nYou can remove this warning by passing 'verification_mode={verification_mode.value}' instead.", FutureWarning)
        if use_auth_token != 'deprecated':
            warnings.warn("'use_auth_token' was deprecated in version 2.7.1 and will be removed in 3.0.0. Pass `token` to `load_dataset_builder` instead.", FutureWarning)
            token = use_auth_token
        else:
            token = self.token
        output_dir = output_dir if output_dir is not None else self._cache_dir
        fs, _, [output_dir] = fsspec.get_fs_token_paths(output_dir, storage_options=storage_options)
        self._fs = fs
        self._output_dir = output_dir if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(output_dir)
        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        base_path = base_path if base_path is not None else self.base_path
        if file_format is not None and file_format not in ['arrow', 'parquet']:
            raise ValueError(f"Unsupported file_format: {file_format}. Expected 'arrow' or 'parquet'")
        self._file_format = file_format
        if self._fs._strip_protocol(self._output_dir) == '':
            raise RuntimeError(f"Unable to download and prepare the dataset at the root {self._output_dir}. Please specify a subdirectory, e.g. '{self._output_dir + self.dataset_name}'")
        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(cache_dir=self._cache_downloaded_dir, force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD, force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD, use_etag=False, num_proc=num_proc, token=token, storage_options=self.storage_options)
            dl_manager = DownloadManager(dataset_name=self.dataset_name, download_config=download_config, data_dir=self.config.data_dir, base_path=base_path, record_checksums=self._record_infos or verification_mode == VerificationMode.ALL_CHECKS)
        is_local = not is_remote_filesystem(self._fs)
        if isinstance(dl_manager, MockDownloadManager) or not is_local or file_format != 'arrow' or (max_shard_size is not None):
            try_from_hf_gcs = False
        self.dl_manager = dl_manager
        if is_local:
            Path(self._output_dir).parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._output_dir + '_builder.lock'
        with FileLock(lock_path) if is_local else contextlib.nullcontext():
            data_exists = self._fs.exists(posixpath.join(self._output_dir, config.DATASET_INFO_FILENAME))
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                logger.info(f'Found cached dataset {self.dataset_name} ({self._output_dir})')
                self.info = self._load_info()
                self.download_post_processing_resources(dl_manager)
                return
            logger.info(f'Generating dataset {self.dataset_name} ({self._output_dir})')
            if is_local:
                if not has_sufficient_disk_space(self.info.size_in_bytes or 0, directory=Path(self._output_dir).parent):
                    raise OSError(f'Not enough disk space. Needed: {size_str(self.info.size_in_bytes or 0)} (download: {size_str(self.info.download_size or 0)}, generated: {size_str(self.info.dataset_size or 0)}, post-processed: {size_str(self.info.post_processing_size or 0)})')

            @contextlib.contextmanager
            def incomplete_dir(dirname):
                """Create temporary dir for dirname and rename on exit."""
                if not is_local:
                    self._fs.makedirs(dirname, exist_ok=True)
                    yield dirname
                else:
                    tmp_dir = dirname + '.incomplete'
                    os.makedirs(tmp_dir, exist_ok=True)
                    try:
                        yield tmp_dir
                        if os.path.isdir(dirname):
                            shutil.rmtree(dirname)
                        shutil.move(tmp_dir, dirname)
                    finally:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)
            if self.info.size_in_bytes:
                logger.info(f'Downloading and preparing dataset {self.dataset_name}/{self.config.name} (download: {size_str(self.info.download_size)}, generated: {size_str(self.info.dataset_size)}, post-processed: {size_str(self.info.post_processing_size)}, total: {size_str(self.info.size_in_bytes)}) to {self._output_dir}...')
            else:
                _dest = self._fs._strip_protocol(self._output_dir) if is_local else self._output_dir
                logger.info(f'Downloading and preparing dataset {self.dataset_name}/{self.config.name} to {_dest}...')
            self._check_manual_download(dl_manager)
            with incomplete_dir(self._output_dir) as tmp_output_dir:
                with temporary_assignment(self, '_output_dir', tmp_output_dir):
                    downloaded_from_gcs = False
                    if try_from_hf_gcs:
                        try:
                            self._download_prepared_from_hf_gcs(dl_manager.download_config)
                            downloaded_from_gcs = True
                        except (DatasetNotOnHfGcsError, MissingFilesOnHfGcsError):
                            logger.info('Dataset not on Hf google storage. Downloading and preparing it from source')
                        except ConnectionError:
                            logger.warning('HF google storage unreachable. Downloading and preparing it from source')
                    if not downloaded_from_gcs:
                        prepare_split_kwargs = {'file_format': file_format}
                        if max_shard_size is not None:
                            prepare_split_kwargs['max_shard_size'] = max_shard_size
                        if num_proc is not None:
                            prepare_split_kwargs['num_proc'] = num_proc
                        self._download_and_prepare(dl_manager=dl_manager, verification_mode=verification_mode, **prepare_split_kwargs, **download_and_prepare_kwargs)
                    self.info.dataset_size = sum((split.num_bytes for split in self.info.splits.values()))
                    self.info.download_checksums = dl_manager.get_recorded_sizes_checksums()
                    self.info.size_in_bytes = self.info.dataset_size + self.info.download_size
                    self._save_info()
            self.download_post_processing_resources(dl_manager)
            logger.info(f'Dataset {self.dataset_name} downloaded and prepared to {self._output_dir}. Subsequent calls will reuse this data.')

    def _check_manual_download(self, dl_manager):
        if self.manual_download_instructions is not None and dl_manager.manual_dir is None:
            raise ManualDownloadError(textwrap.dedent(f'                    The dataset {self.dataset_name} with config {self.config.name} requires manual data.\n                    Please follow the manual download instructions:\n                     {self.manual_download_instructions}\n                    Manual data can be loaded with:\n                     datasets.load_dataset("{self.dataset_name}", data_dir="<path/to/manual/data>")'))

    def _download_prepared_from_hf_gcs(self, download_config: DownloadConfig):
        relative_data_dir = self._relative_data_dir(with_version=True, with_hash=False)
        reader = ArrowReader(self._output_dir, self.info)
        reader.download_from_hf_gcs(download_config, relative_data_dir)
        downloaded_info = DatasetInfo.from_directory(self._output_dir)
        self.info.update(downloaded_info)
        remote_cache_dir = HF_GCP_BASE_URL + '/' + relative_data_dir.replace(os.sep, '/')
        for split in self.info.splits:
            for resource_file_name in self._post_processing_resources(split).values():
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
                try:
                    resource_path = cached_path(remote_cache_dir + '/' + resource_file_name)
                    shutil.move(resource_path, os.path.join(self._output_dir, resource_file_name))
                except ConnectionError:
                    logger.info(f"Couldn't download resourse file {resource_file_name} from Hf google storage.")
        logger.info('Dataset downloaded from Hf google storage.')

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs):
        """Downloads and prepares dataset for reading.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required data and generate
        the pre-processed datasets files.

        Args:
            dl_manager ([`DownloadManager`]):
                `DownloadManager` used to download and cache data.
            verification_mode ([`VerificationMode`]):
                if `ALL_CHECKS`, perform all the verifications including checksums.
                if `BASIC_CHECKS`, do not perform checksums, only perform split tests.
                if `NO_CHECKS`, do not perform any verification.
            prepare_split_kwargs: Additional options, such as `file_format`, `max_shard_size`
        """
        split_dict = SplitDict(dataset_name=self.dataset_name)
        split_generators_kwargs = self._make_split_generators_kwargs(prepare_split_kwargs)
        split_generators = self._split_generators(dl_manager, **split_generators_kwargs)
        if verification_mode == VerificationMode.ALL_CHECKS and dl_manager.record_checksums:
            verify_checksums(self.info.download_checksums, dl_manager.get_recorded_sizes_checksums(), 'dataset source files')
        for split_generator in split_generators:
            if str(split_generator.split_info.name).lower() == 'all':
                raise ValueError('`all` is a special split keyword corresponding to the union of all splits, so cannot be used as key in ._split_generator().')
            logger.info(f'Generating {split_generator.split_info.name} split')
            split_dict.add(split_generator.split_info)
            try:
                self._prepare_split(split_generator, **prepare_split_kwargs)
            except OSError as e:
                raise OSError('Cannot find data file. ' + (self.manual_download_instructions or '') + '\nOriginal error:\n' + str(e)) from None
            except DuplicatedKeysError as e:
                raise DuplicatedKeysError(e.key, e.duplicate_key_indices, fix_msg=f'To avoid duplicate keys, please fix the dataset script {self.name}.py') from None
            dl_manager.manage_extracted_files()
        if verification_mode == VerificationMode.BASIC_CHECKS or verification_mode == VerificationMode.ALL_CHECKS:
            verify_splits(self.info.splits, split_dict)
        self.info.splits = split_dict
        self.info.download_size = dl_manager.downloaded_size

    def download_post_processing_resources(self, dl_manager):
        for split in self.info.splits or []:
            for resource_name, resource_file_name in self._post_processing_resources(split).items():
                if not not is_remote_filesystem(self._fs):
                    raise NotImplementedError(f'Post processing is not supported on filesystem {self._fs}')
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
                resource_path = os.path.join(self._output_dir, resource_file_name)
                if not os.path.exists(resource_path):
                    downloaded_resource_path = self._download_post_processing_resources(split, resource_name, dl_manager)
                    if downloaded_resource_path:
                        logger.info(f'Downloaded post-processing resource {resource_name} as {resource_file_name}')
                        shutil.move(downloaded_resource_path, resource_path)

    def _load_info(self) -> DatasetInfo:
        return DatasetInfo.from_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_info(self):
        file_lock = FileLock(self._output_dir + '_info.lock') if not is_remote_filesystem(self._fs) else contextlib.nullcontext()
        with file_lock:
            self.info.write_to_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_infos(self):
        file_lock = FileLock(self._output_dir + '_infos.lock') if not is_remote_filesystem(self._fs) else contextlib.nullcontext()
        with file_lock:
            DatasetInfosDict(**{self.config.name: self.info}).write_to_directory(self.get_imported_module_dir())

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        """Get kwargs for `self._split_generators()` from `prepare_split_kwargs`."""
        del prepare_split_kwargs
        return {}

    def as_dataset(self, split: Optional[Split]=None, run_post_process=True, verification_mode: Optional[Union[VerificationMode, str]]=None, ignore_verifications='deprecated', in_memory=False) -> Union[Dataset, DatasetDict]:
        """Return a Dataset for the specified split.

        Args:
            split (`datasets.Split`):
                Which subset of the data to return.
            run_post_process (`bool`, defaults to `True`):
                Whether to run post-processing dataset transforms and/or add
                indexes.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Whether to ignore the verifications of the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            datasets.Dataset

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder('rotten_tomatoes')
        >>> builder.download_and_prepare()
        >>> ds = builder.as_dataset(split='train')
        >>> ds
        Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        ```
        """
        if ignore_verifications != 'deprecated':
            verification_mode = verification_mode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(f"'ignore_verifications' was deprecated in favor of 'verification' in version 2.9.1 and will be removed in 3.0.0.\nYou can remove this warning by passing 'verification_mode={verification_mode.value}' instead.", FutureWarning)
        if self._file_format is not None and self._file_format != 'arrow':
            raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f'Loading a dataset cached in a {type(self._fs).__name__} is not supported.')
        if not os.path.exists(self._output_dir):
            raise FileNotFoundError(f'Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call builder.download_and_prepare(), or use datasets.load_dataset() before trying to access the Dataset object.')
        logger.debug(f'Constructing Dataset for split {split or ', '.join(self.info.splits)}, from {self._output_dir}')
        if split is None:
            split = {s: s for s in self.info.splits}
        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        datasets = map_nested(partial(self._build_single_dataset, run_post_process=run_post_process, verification_mode=verification_mode, in_memory=in_memory), split, map_tuple=True, disable_tqdm=True)
        if isinstance(datasets, dict):
            datasets = DatasetDict(datasets)
        return datasets

    def _build_single_dataset(self, split: Union[str, ReadInstruction, Split], run_post_process: bool, verification_mode: VerificationMode, in_memory: bool=False):
        """as_dataset for a single split."""
        if not isinstance(split, ReadInstruction):
            split = str(split)
            if split == 'all':
                split = '+'.join(self.info.splits.keys())
            split = Split(split)
        ds = self._as_dataset(split=split, in_memory=in_memory)
        if run_post_process:
            for resource_file_name in self._post_processing_resources(split).values():
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
            resources_paths = {resource_name: os.path.join(self._output_dir, resource_file_name) for resource_name, resource_file_name in self._post_processing_resources(split).items()}
            post_processed = self._post_process(ds, resources_paths)
            if post_processed is not None:
                ds = post_processed
                recorded_checksums = {}
                record_checksums = False
                for resource_name, resource_path in resources_paths.items():
                    size_checksum = get_size_checksum_dict(resource_path)
                    recorded_checksums[resource_name] = size_checksum
                if verification_mode == VerificationMode.ALL_CHECKS and record_checksums:
                    if self.info.post_processed is None or self.info.post_processed.resources_checksums is None:
                        expected_checksums = None
                    else:
                        expected_checksums = self.info.post_processed.resources_checksums.get(split)
                    verify_checksums(expected_checksums, recorded_checksums, 'post processing resources')
                if self.info.post_processed is None:
                    self.info.post_processed = PostProcessedInfo()
                if self.info.post_processed.resources_checksums is None:
                    self.info.post_processed.resources_checksums = {}
                self.info.post_processed.resources_checksums[str(split)] = recorded_checksums
                self.info.post_processing_size = sum((checksums_dict['num_bytes'] for split_checksums_dicts in self.info.post_processed.resources_checksums.values() for checksums_dict in split_checksums_dicts.values()))
                if self.info.dataset_size is not None and self.info.download_size is not None:
                    self.info.size_in_bytes = self.info.dataset_size + self.info.download_size + self.info.post_processing_size
                self._save_info()
                ds._info.post_processed = self.info.post_processed
                ds._info.post_processing_size = self.info.post_processing_size
                ds._info.size_in_bytes = self.info.size_in_bytes
                if self.info.post_processed.features is not None:
                    if self.info.post_processed.features.type != ds.features.type:
                        raise ValueError(f"Post-processed features info don't match the dataset:\nGot\n{self.info.post_processed.features}\nbut expected something like\n{ds.features}")
                    else:
                        ds.info.features = self.info.post_processed.features
        return ds

    def _as_dataset(self, split: Union[ReadInstruction, Split]=Split.TRAIN, in_memory: bool=False) -> Dataset:
        """Constructs a `Dataset`.

        This is the internal implementation to overwrite called when user calls
        `as_dataset`. It should read the pre-processed datasets files and generate
        the `Dataset` object.

        Args:
            split (`datasets.Split`):
                which subset of the data to read.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            `Dataset`
        """
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_name = self.dataset_name
        if self._check_legacy_cache():
            dataset_name = self.name
        dataset_kwargs = ArrowReader(cache_dir, self.info).read(name=dataset_name, instructions=split, split_infos=self.info.splits.values(), in_memory=in_memory)
        fingerprint = self._get_dataset_fingerprint(split)
        return Dataset(fingerprint=fingerprint, **dataset_kwargs)

    def _get_dataset_fingerprint(self, split: Union[ReadInstruction, Split]) -> str:
        """The dataset fingerprint is the hash of the relative directory dataset_name/config_name/version/hash, as well as the split specs."""
        hasher = Hasher()
        hasher.update(Path(self._relative_data_dir()).as_posix())
        hasher.update(str(split))
        fingerprint = hasher.hexdigest()
        return fingerprint

    def as_streaming_dataset(self, split: Optional[str]=None, base_path: Optional[str]=None) -> Union[Dict[str, IterableDataset], IterableDataset]:
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f'Loading a streaming dataset cached in a {type(self._fs).__name__} is not supported yet.')
        dl_manager = StreamingDownloadManager(base_path=base_path or self.base_path, download_config=DownloadConfig(token=self.token, storage_options=self.storage_options), dataset_name=self.dataset_name, data_dir=self.config.data_dir)
        self._check_manual_download(dl_manager)
        splits_generators = {sg.name: sg for sg in self._split_generators(dl_manager)}
        if split is None:
            splits_generator = splits_generators
        elif split in splits_generators:
            splits_generator = splits_generators[split]
        else:
            raise ValueError(f'Bad split: {split}. Available splits: {list(splits_generators)}')
        datasets = map_nested(self._as_streaming_dataset_single, splits_generator, map_tuple=True)
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _as_streaming_dataset_single(self, splits_generator) -> IterableDataset:
        ex_iterable = self._get_examples_iterable_for_split(splits_generator)
        token_per_repo_id = {self.repo_id: self.token} if self.repo_id else {}
        return IterableDataset(ex_iterable, info=self.info, split=splits_generator.name, token_per_repo_id=token_per_repo_id)

    def _post_process(self, dataset: Dataset, resources_paths: Mapping[str, str]) -> Optional[Dataset]:
        """Run dataset transforms or add indexes"""
        return None

    def _post_processing_resources(self, split: str) -> Dict[str, str]:
        """Mapping resource_name -> resource_file_name"""
        return {}

    def _download_post_processing_resources(self, split: str, resource_name: str, dl_manager: DownloadManager) -> Optional[str]:
        """Download the resource using the download manager and return the downloaded path."""
        return None

    @abc.abstractmethod
    def _split_generators(self, dl_manager: DownloadManager):
        """Specify feature dictionary generators and dataset splits.

        This function returns a list of `SplitGenerator`s defining how to generate
        data and what splits to use.

        Example:

            return [
                    datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={'file': 'train_data.zip'},
                    ),
                    datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={'file': 'test_data.zip'},
                    ),
            ]

        The above code will first call `_generate_examples(file='train_data.zip')`
        to write the train data, then `_generate_examples(file='test_data.zip')` to
        write the test data.

        Datasets are typically split into different subsets to be used at various
        stages of training and evaluation.

        Note that for datasets without a `VALIDATION` split, you can use a
        fraction of the `TRAIN` data for evaluation as you iterate on your model
        so as not to overfit to the `TEST` data.

        For downloads and extractions, use the given `download_manager`.
        Note that the `DownloadManager` caches downloads, so it is fine to have each
        generator attempt to download the source data.

        A good practice is to download all data in this function, and then
        distribute the relevant parts to each split with the `gen_kwargs` argument

        Args:
            dl_manager (`DownloadManager`):
                Download manager to download the data

        Returns:
            `list<SplitGenerator>`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _prepare_split(self, split_generator: SplitGenerator, file_format: str='arrow', max_shard_size: Optional[Union[str, int]]=None, num_proc: Optional[int]=None, **kwargs):
        """Generate the examples and record them on disk.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
            file_format (`str`, *optional*):
                format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            **kwargs: Additional kwargs forwarded from _download_and_prepare (ex:
                beam pipeline)
        """
        raise NotImplementedError()

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        """Generate the examples on the fly.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
        """
        raise NotImplementedError()