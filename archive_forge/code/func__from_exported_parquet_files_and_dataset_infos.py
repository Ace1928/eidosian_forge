import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import yaml
from huggingface_hub import DatasetCardData
from ..config import METADATA_CONFIGS_FIELD
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger
from .deprecation_utils import deprecated
@classmethod
def _from_exported_parquet_files_and_dataset_infos(cls, revision: str, exported_parquet_files: List[Dict[str, Any]], dataset_infos: DatasetInfosDict) -> 'MetadataConfigs':
    metadata_configs = {config_name: {'data_files': [{'split': split_name, 'path': [parquet_file['url'].replace('refs%2Fconvert%2Fparquet', revision) for parquet_file in parquet_files_for_split]} for split_name, parquet_files_for_split in groupby(parquet_files_for_config, itemgetter('split'))], 'version': str(dataset_infos.get(config_name, DatasetInfo()).version or '0.0.0')} for config_name, parquet_files_for_config in groupby(exported_parquet_files, itemgetter('config'))}
    if dataset_infos:
        metadata_configs = {config_name: {'data_files': [data_file for split_name in dataset_info.splits for data_file in metadata_configs[config_name]['data_files'] if data_file['split'] == split_name], 'version': metadata_configs[config_name]['version']} for config_name, dataset_info in dataset_infos.items()}
    return cls(metadata_configs)