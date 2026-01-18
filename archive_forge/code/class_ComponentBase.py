from __future__ import annotations
import abc
import hashlib
import json
import sys
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import gradio_client.utils as client_utils
from gradio import utils
from gradio.blocks import Block, BlockContext
from gradio.component_meta import ComponentMeta
from gradio.data_classes import GradioDataModel
from gradio.events import EventListener
from gradio.layouts import Form
from gradio.processing_utils import move_files_to_cache
class ComponentBase(ABC, metaclass=ComponentMeta):
    EVENTS: list[EventListener | str] = []

    @abstractmethod
    def preprocess(self, payload: Any) -> Any:
        """
        Any preprocessing needed to be performed on function input.
        Parameters:
            payload: The input data received by the component from the frontend.
        Returns:
            The preprocessed input data sent to the user's function in the backend.
        """
        return payload

    @abstractmethod
    def postprocess(self, value):
        """
        Any postprocessing needed to be performed on function output.
        Parameters:
            value: The output data received by the component from the user's function in the backend.
        Returns:
            The postprocessed output data sent to the frontend.
        """
        return value

    @abstractmethod
    def process_example(self, value):
        """
        Process the input data in a way that can be displayed by the examples dataset component in the front-end.

        For example, only return the name of a file as opposed to a full path. Or get the head of a dataframe.
        The return value must be able to be json-serializable to put in the config.
        """
        pass

    @abstractmethod
    def api_info(self) -> dict[str, list[str]]:
        """
        The typing information for this component as a dictionary whose values are a list of 2 strings: [Python type, language-agnostic description].
        Keys of the dictionary are: raw_input, raw_output, serialized_input, serialized_output
        """
        pass

    @abstractmethod
    def example_inputs(self) -> Any:
        """
        Deprecated and replaced by `example_payload()` and `example_value()`.
        """
        pass

    @abstractmethod
    def flag(self, payload: Any | GradioDataModel, flag_dir: str | Path='') -> str:
        """
        Write the component's value to a format that can be stored in a csv or jsonl format for flagging.
        """
        pass

    @abstractmethod
    def read_from_flag(self, payload: Any) -> GradioDataModel | Any:
        """
        Convert the data from the csv or jsonl file into the component state.
        """
        return payload

    @property
    @abstractmethod
    def skip_api(self):
        """Whether this component should be skipped from the api return value"""

    @classmethod
    def has_event(cls, event: str | EventListener) -> bool:
        return event in cls.EVENTS

    @classmethod
    def get_component_class_id(cls) -> str:
        module_name = cls.__module__
        module_path = sys.modules[module_name].__file__
        module_hash = hashlib.md5(f'{cls.__name__}_{module_path}'.encode()).hexdigest()
        return module_hash