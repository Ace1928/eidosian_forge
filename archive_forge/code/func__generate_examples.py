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
def _generate_examples(self, files, metadata_files, split_name, add_metadata, add_labels):
    split_metadata_files = metadata_files.get(split_name, [])
    sample_empty_metadata = {k: None for k in self.info.features if k != self.BASE_COLUMN_NAME} if self.info.features else {}
    last_checked_dir = None
    metadata_dir = None
    metadata_dict = None
    downloaded_metadata_file = None
    metadata_ext = ''
    if split_metadata_files:
        metadata_ext = {os.path.splitext(original_metadata_file)[-1] for original_metadata_file, _ in split_metadata_files}
        metadata_ext = metadata_ext.pop()
    file_idx = 0
    for original_file, downloaded_file_or_dir in files:
        if original_file is not None:
            _, original_file_ext = os.path.splitext(original_file)
            if original_file_ext.lower() in self.EXTENSIONS:
                if add_metadata:
                    current_dir = os.path.dirname(original_file)
                    if last_checked_dir is None or last_checked_dir != current_dir:
                        last_checked_dir = current_dir
                        metadata_file_candidates = [(os.path.relpath(original_file, os.path.dirname(metadata_file_candidate)), metadata_file_candidate, downloaded_metadata_file) for metadata_file_candidate, downloaded_metadata_file in split_metadata_files if metadata_file_candidate is not None and (not os.path.relpath(original_file, os.path.dirname(metadata_file_candidate)).startswith('..'))]
                        if metadata_file_candidates:
                            _, metadata_file, downloaded_metadata_file = min(metadata_file_candidates, key=lambda x: count_path_segments(x[0]))
                            pa_metadata_table = self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext)
                            pa_file_name_array = pa_metadata_table['file_name']
                            pa_metadata_table = pa_metadata_table.drop(['file_name'])
                            metadata_dir = os.path.dirname(metadata_file)
                            metadata_dict = {os.path.normpath(file_name).replace('\\', '/'): sample_metadata for file_name, sample_metadata in zip(pa_file_name_array.to_pylist(), pa_metadata_table.to_pylist())}
                        else:
                            raise ValueError(f'One or several metadata{metadata_ext} were found, but not in the same directory or in a parent directory of {downloaded_file_or_dir}.')
                    if metadata_dir is not None and downloaded_metadata_file is not None:
                        file_relpath = os.path.relpath(original_file, metadata_dir)
                        file_relpath = file_relpath.replace('\\', '/')
                        if file_relpath not in metadata_dict:
                            raise ValueError(f"{self.BASE_COLUMN_NAME} at {file_relpath} doesn't have metadata in {downloaded_metadata_file}.")
                        sample_metadata = metadata_dict[file_relpath]
                    else:
                        raise ValueError(f'One or several metadata{metadata_ext} were found, but not in the same directory or in a parent directory of {downloaded_file_or_dir}.')
                else:
                    sample_metadata = {}
                if add_labels:
                    sample_label = {'label': os.path.basename(os.path.dirname(original_file))}
                else:
                    sample_label = {}
                yield (file_idx, {**sample_empty_metadata, self.BASE_COLUMN_NAME: downloaded_file_or_dir, **sample_metadata, **sample_label})
                file_idx += 1
        else:
            for downloaded_dir_file in downloaded_file_or_dir:
                _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                if downloaded_dir_file_ext.lower() in self.EXTENSIONS:
                    if add_metadata:
                        current_dir = os.path.dirname(downloaded_dir_file)
                        if last_checked_dir is None or last_checked_dir != current_dir:
                            last_checked_dir = current_dir
                            metadata_file_candidates = [(os.path.relpath(downloaded_dir_file, os.path.dirname(downloaded_metadata_file)), metadata_file_candidate, downloaded_metadata_file) for metadata_file_candidate, downloaded_metadata_file in split_metadata_files if metadata_file_candidate is None and (not os.path.relpath(downloaded_dir_file, os.path.dirname(downloaded_metadata_file)).startswith('..'))]
                            if metadata_file_candidates:
                                _, metadata_file, downloaded_metadata_file = min(metadata_file_candidates, key=lambda x: count_path_segments(x[0]))
                                pa_metadata_table = self._read_metadata(downloaded_metadata_file, metadata_ext=metadata_ext)
                                pa_file_name_array = pa_metadata_table['file_name']
                                pa_metadata_table = pa_metadata_table.drop(['file_name'])
                                metadata_dir = os.path.dirname(downloaded_metadata_file)
                                metadata_dict = {os.path.normpath(file_name).replace('\\', '/'): sample_metadata for file_name, sample_metadata in zip(pa_file_name_array.to_pylist(), pa_metadata_table.to_pylist())}
                            else:
                                raise ValueError(f'One or several metadata{metadata_ext} were found, but not in the same directory or in a parent directory of {downloaded_dir_file}.')
                        if metadata_dir is not None and downloaded_metadata_file is not None:
                            downloaded_dir_file_relpath = os.path.relpath(downloaded_dir_file, metadata_dir)
                            downloaded_dir_file_relpath = downloaded_dir_file_relpath.replace('\\', '/')
                            if downloaded_dir_file_relpath not in metadata_dict:
                                raise ValueError(f"{self.BASE_COLUMN_NAME} at {downloaded_dir_file_relpath} doesn't have metadata in {downloaded_metadata_file}.")
                            sample_metadata = metadata_dict[downloaded_dir_file_relpath]
                        else:
                            raise ValueError(f'One or several metadata{metadata_ext} were found, but not in the same directory or in a parent directory of {downloaded_dir_file}.')
                    else:
                        sample_metadata = {}
                    if add_labels:
                        sample_label = {'label': os.path.basename(os.path.dirname(downloaded_dir_file))}
                    else:
                        sample_label = {}
                    yield (file_idx, {**sample_empty_metadata, self.BASE_COLUMN_NAME: downloaded_dir_file, **sample_metadata, **sample_label})
                    file_idx += 1