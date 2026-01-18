import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
@dataclass
class DatasetInfo:
    """Information about a dataset.

    `DatasetInfo` documents datasets, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Not all fields are known on construction and may be updated later.

    Attributes:
        description (`str`):
            A description of the dataset.
        citation (`str`):
            A BibTeX citation of the dataset.
        homepage (`str`):
            A URL to the official homepage for the dataset.
        license (`str`):
            The dataset's license. It can be the name of the license or a paragraph containing the terms of the license.
        features ([`Features`], *optional*):
            The features used to specify the dataset's column types.
        post_processed (`PostProcessedInfo`, *optional*):
            Information regarding the resources of a possible post-processing of a dataset. For example, it can contain the information of an index.
        supervised_keys (`SupervisedKeysData`, *optional*):
            Specifies the input feature and the label for supervised learning if applicable for the dataset (legacy from TFDS).
        builder_name (`str`, *optional*):
            The name of the `GeneratorBasedBuilder` subclass used to create the dataset. Usually matched to the corresponding script name. It is also the snake_case version of the dataset builder class name.
        config_name (`str`, *optional*):
            The name of the configuration derived from [`BuilderConfig`].
        version (`str` or [`Version`], *optional*):
            The version of the dataset.
        splits (`dict`, *optional*):
            The mapping between split name and metadata.
        download_checksums (`dict`, *optional*):
            The mapping between the URL to download the dataset's checksums and corresponding metadata.
        download_size (`int`, *optional*):
            The size of the files to download to generate the dataset, in bytes.
        post_processing_size (`int`, *optional*):
            Size of the dataset in bytes after post-processing, if any.
        dataset_size (`int`, *optional*):
            The combined size in bytes of the Arrow tables for all splits.
        size_in_bytes (`int`, *optional*):
            The combined size in bytes of all files associated with the dataset (downloaded files + Arrow files).
        task_templates (`List[TaskTemplate]`, *optional*):
            The task templates to prepare the dataset for during training and evaluation. Each template casts the dataset's [`Features`] to standardized column names and types as detailed in `datasets.tasks`.
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`] and used in the [`DatasetBuilder`].
    """
    description: str = dataclasses.field(default_factory=str)
    citation: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    features: Optional[Features] = None
    post_processed: Optional[PostProcessedInfo] = None
    supervised_keys: Optional[SupervisedKeysData] = None
    task_templates: Optional[List[TaskTemplate]] = None
    builder_name: Optional[str] = None
    dataset_name: Optional[str] = None
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    splits: Optional[dict] = None
    download_checksums: Optional[dict] = None
    download_size: Optional[int] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None
    _INCLUDED_INFO_IN_YAML: ClassVar[List[str]] = ['config_name', 'download_size', 'dataset_size', 'features', 'splits']

    def __post_init__(self):
        if self.features is not None and (not isinstance(self.features, Features)):
            self.features = Features.from_dict(self.features)
        if self.post_processed is not None and (not isinstance(self.post_processed, PostProcessedInfo)):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and (not isinstance(self.version, Version)):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.splits is not None and (not isinstance(self.splits, SplitDict)):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and (not isinstance(self.supervised_keys, SupervisedKeysData)):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)
        if self.task_templates is not None:
            if isinstance(self.task_templates, (list, tuple)):
                templates = [template if isinstance(template, TaskTemplate) else task_template_from_dict(template) for template in self.task_templates]
                self.task_templates = [template for template in templates if template is not None]
            elif isinstance(self.task_templates, TaskTemplate):
                self.task_templates = [self.task_templates]
            else:
                template = task_template_from_dict(self.task_templates)
                self.task_templates = [template] if template is not None else []
        if self.task_templates is not None:
            self.task_templates = list(self.task_templates)
            if self.features is not None:
                self.task_templates = [template.align_with_features(self.features) for template in self.task_templates]

    def write_to_directory(self, dataset_info_dir, pretty_print=False, fs='deprecated', storage_options: Optional[dict]=None):
        """Write `DatasetInfo` and license (if present) as JSON files to `dataset_info_dir`.

        Args:
            dataset_info_dir (`str`):
                Destination directory.
            pretty_print (`bool`, defaults to `False`):
                If `True`, the JSON will be pretty-printed with the indent level of 4.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem used to download the files from.

                <Deprecated version="2.9.0">

                `fs` was deprecated in version 2.9.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`.

                </Deprecated>

            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.info.write_to_directory("/path/to/directory/")
        ```
        """
        if fs != 'deprecated':
            warnings.warn("'fs' was deprecated in favor of 'storage_options' in version 2.9.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'storage_options=fs.storage_options' instead.", FutureWarning)
            storage_options = fs.storage_options
        fs: fsspec.AbstractFileSystem
        fs, _, _ = fsspec.get_fs_token_paths(dataset_info_dir, storage_options=storage_options)
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), 'wb') as f:
            self._dump_info(f, pretty_print=pretty_print)
        if self.license:
            with fs.open(posixpath.join(dataset_info_dir, config.LICENSE_FILENAME), 'wb') as f:
                self._dump_license(f)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(json.dumps(asdict(self), indent=4 if pretty_print else None).encode('utf-8'))

    def _dump_license(self, file):
        """Dump license in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(self.license.encode('utf-8'))

    @classmethod
    def from_merge(cls, dataset_infos: List['DatasetInfo']):
        dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]
        if len(dataset_infos) > 0 and all((dataset_infos[0] == dset_info for dset_info in dataset_infos)):
            return dataset_infos[0]
        description = '\n\n'.join(unique_values((info.description for info in dataset_infos))).strip()
        citation = '\n\n'.join(unique_values((info.citation for info in dataset_infos))).strip()
        homepage = '\n\n'.join(unique_values((info.homepage for info in dataset_infos))).strip()
        license = '\n\n'.join(unique_values((info.license for info in dataset_infos))).strip()
        features = None
        supervised_keys = None
        task_templates = None
        all_task_templates = [info.task_templates for info in dataset_infos if info.task_templates is not None]
        if len(all_task_templates) > 1:
            task_templates = list(set(all_task_templates[0]).intersection(*all_task_templates[1:]))
        elif len(all_task_templates):
            task_templates = list(set(all_task_templates[0]))
        task_templates = task_templates if task_templates else None
        return cls(description=description, citation=citation, homepage=homepage, license=license, features=features, supervised_keys=supervised_keys, task_templates=task_templates)

    @classmethod
    def from_directory(cls, dataset_info_dir: str, fs='deprecated', storage_options: Optional[dict]=None) -> 'DatasetInfo':
        """Create [`DatasetInfo`] from the JSON file in `dataset_info_dir`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the [`DatasetInfo`].

        This will overwrite all previous metadata.

        Args:
            dataset_info_dir (`str`):
                The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem used to download the files from.

                <Deprecated version="2.9.0">

                `fs` was deprecated in version 2.9.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`.

                </Deprecated>

            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import DatasetInfo
        >>> ds_info = DatasetInfo.from_directory("/path/to/directory/")
        ```
        """
        if fs != 'deprecated':
            warnings.warn("'fs' was deprecated in favor of 'storage_options' in version 2.9.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'storage_options=fs.storage_options' instead.", FutureWarning)
            storage_options = fs.storage_options
        fs: fsspec.AbstractFileSystem
        fs, _, _ = fsspec.get_fs_token_paths(dataset_info_dir, storage_options=storage_options)
        logger.info(f'Loading Dataset info from {dataset_info_dir}')
        if not dataset_info_dir:
            raise ValueError('Calling DatasetInfo.from_directory() with undefined dataset_info_dir.')
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), 'r', encoding='utf-8') as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> 'DatasetInfo':
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: 'DatasetInfo', ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(**{k: copy.deepcopy(v) for k, v in other_dataset_info.__dict__.items() if v is not None or not ignore_none})

    def copy(self) -> 'DatasetInfo':
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def _to_yaml_dict(self) -> dict:
        yaml_dict = {}
        dataset_info_dict = asdict(self)
        for key in dataset_info_dict:
            if key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, '_to_yaml_list'):
                    yaml_dict[key] = value._to_yaml_list()
                elif hasattr(value, '_to_yaml_string'):
                    yaml_dict[key] = value._to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def _from_yaml_dict(cls, yaml_data: dict) -> 'DatasetInfo':
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get('features') is not None:
            yaml_data['features'] = Features._from_yaml_list(yaml_data['features'])
        if yaml_data.get('splits') is not None:
            yaml_data['splits'] = SplitDict._from_yaml_list(yaml_data['splits'])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})