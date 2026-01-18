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
class DatasetInfosDict(Dict[str, DatasetInfo]):

    def write_to_directory(self, dataset_infos_dir, overwrite=False, pretty_print=False) -> None:
        total_dataset_infos = {}
        dataset_infos_path = os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)
        dataset_readme_path = os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)
        if not overwrite:
            total_dataset_infos = self.from_directory(dataset_infos_dir)
        total_dataset_infos.update(self)
        if os.path.exists(dataset_infos_path):
            with open(dataset_infos_path, 'w', encoding='utf-8') as f:
                dataset_infos_dict = {config_name: asdict(dset_info) for config_name, dset_info in total_dataset_infos.items()}
                json.dump(dataset_infos_dict, f, indent=4 if pretty_print else None)
        if os.path.exists(dataset_readme_path):
            dataset_card = DatasetCard.load(dataset_readme_path)
            dataset_card_data = dataset_card.data
        else:
            dataset_card = None
            dataset_card_data = DatasetCardData()
        if total_dataset_infos:
            total_dataset_infos.to_dataset_card_data(dataset_card_data)
            dataset_card = DatasetCard('---\n' + str(dataset_card_data) + '\n---\n') if dataset_card is None else dataset_card
            dataset_card.save(Path(dataset_readme_path))

    @classmethod
    def from_directory(cls, dataset_infos_dir) -> 'DatasetInfosDict':
        logger.info(f'Loading Dataset Infos from {dataset_infos_dir}')
        if os.path.exists(os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)):
            dataset_card_data = DatasetCard.load(Path(dataset_infos_dir) / config.REPOCARD_FILENAME).data
            if 'dataset_info' in dataset_card_data:
                return cls.from_dataset_card_data(dataset_card_data)
        if os.path.exists(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME), encoding='utf-8') as f:
                return cls({config_name: DatasetInfo.from_dict(dataset_info_dict) for config_name, dataset_info_dict in json.load(f).items()})
        else:
            return cls()

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> 'DatasetInfosDict':
        if isinstance(dataset_card_data.get('dataset_info'), (list, dict)):
            if isinstance(dataset_card_data['dataset_info'], list):
                return cls({dataset_info_yaml_dict.get('config_name', 'default'): DatasetInfo._from_yaml_dict(dataset_info_yaml_dict) for dataset_info_yaml_dict in dataset_card_data['dataset_info']})
            else:
                dataset_info = DatasetInfo._from_yaml_dict(dataset_card_data['dataset_info'])
                dataset_info.config_name = dataset_card_data['dataset_info'].get('config_name', 'default')
                return cls({dataset_info.config_name: dataset_info})
        else:
            return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            if 'dataset_info' in dataset_card_data and isinstance(dataset_card_data['dataset_info'], dict):
                dataset_metadata_infos = {dataset_card_data['dataset_info'].get('config_name', 'default'): dataset_card_data['dataset_info']}
            elif 'dataset_info' in dataset_card_data and isinstance(dataset_card_data['dataset_info'], list):
                dataset_metadata_infos = {config_metadata['config_name']: config_metadata for config_metadata in dataset_card_data['dataset_info']}
            else:
                dataset_metadata_infos = {}
            total_dataset_infos = {**dataset_metadata_infos, **{config_name: dset_info._to_yaml_dict() for config_name, dset_info in self.items()}}
            for config_name, dset_info_yaml_dict in total_dataset_infos.items():
                dset_info_yaml_dict['config_name'] = config_name
            if len(total_dataset_infos) == 1:
                dataset_card_data['dataset_info'] = next(iter(total_dataset_infos.values()))
                config_name = dataset_card_data['dataset_info'].pop('config_name', None)
                if config_name != 'default':
                    dataset_card_data['dataset_info'] = {'config_name': config_name, **dataset_card_data['dataset_info']}
            else:
                dataset_card_data['dataset_info'] = []
                for config_name, dataset_info_yaml_dict in sorted(total_dataset_infos.items()):
                    dataset_info_yaml_dict.pop('config_name', None)
                    dataset_info_yaml_dict = {'config_name': config_name, **dataset_info_yaml_dict}
                    dataset_card_data['dataset_info'].append(dataset_info_yaml_dict)