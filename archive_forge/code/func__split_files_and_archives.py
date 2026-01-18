import collections
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
import pandas as pd
import pyarrow as pa
import pyarrow.json as paj
import datasets
from datasets.features.features import FeatureType
from datasets.tasks.base import TaskTemplate
def _split_files_and_archives(self, data_files):
    files, archives = ([], [])
    for data_file in data_files:
        _, data_file_ext = os.path.splitext(data_file)
        if data_file_ext.lower() in self.EXTENSIONS:
            files.append(data_file)
        elif os.path.basename(data_file) in self.METADATA_FILENAMES:
            files.append(data_file)
        else:
            archives.append(data_file)
    return (files, archives)