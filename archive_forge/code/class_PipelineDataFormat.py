import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    `PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
    pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """
    SUPPORTED_FORMATS = ['json', 'csv', 'pipe']

    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite: bool=False):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(',') if column is not None else ['']
        self.is_multi_columns = len(self.column) > 1
        if self.is_multi_columns:
            self.column = [tuple(c.split('=')) if '=' in c else (c, c) for c in self.column]
        if output_path is not None and (not overwrite):
            if exists(abspath(self.output_path)):
                raise OSError(f'{self.output_path} already exists on disk')
        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError(f'{self.input_path} doesnt exist on disk')

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`dict` or list of `dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (`dict` or list of `dict`): The data to store.

        Returns:
            `str`: Path where the data has been saved.
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, 'pickle'))
        with open(binary_path, 'wb+') as f_output:
            pickle.dump(data, f_output)
        return binary_path

    @staticmethod
    def from_str(format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False) -> 'PipelineDataFormat':
        """
        Creates an instance of the right subclass of [`~pipelines.PipelineDataFormat`] depending on `format`.

        Args:
            format (`str`):
                The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
            output_path (`str`, *optional*):
                Where to save the outgoing data.
            input_path (`str`, *optional*):
                Where to look for the input data.
            column (`str`, *optional*):
                The column to read.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether or not to overwrite the `output_path`.

        Returns:
            [`~pipelines.PipelineDataFormat`]: The proper data format.
        """
        if format == 'json':
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == 'csv':
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == 'pipe':
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError(f'Unknown reader {format} (Available reader are json/csv/pipe)')