import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
class SpaceCardData(CardData):
    """Space Card Metadata that is used by Hugging Face Hub when included at the top of your README.md

    To get an exhaustive reference of Spaces configuration, please visit https://huggingface.co/docs/hub/spaces-config-reference#spaces-configuration-reference.

    Args:
        title (`str`, *optional*)
            Title of the Space.
        sdk (`str`, *optional*)
            SDK of the Space (one of `gradio`, `streamlit`, `docker`, or `static`).
        sdk_version (`str`, *optional*)
            Version of the used SDK (if Gradio/Streamlit sdk).
        python_version (`str`, *optional*)
            Python version used in the Space (if Gradio/Streamlit sdk).
        app_file (`str`, *optional*)
            Path to your main application file (which contains either gradio or streamlit Python code, or static html code).
            Path is relative to the root of the repository.
        app_port (`str`, *optional*)
            Port on which your application is running. Used only if sdk is `docker`.
        license (`str`, *optional*)
            License of this model. Example: apache-2.0 or any license from
            https://huggingface.co/docs/hub/repositories-licenses.
        duplicated_from (`str`, *optional*)
            ID of the original Space if this is a duplicated Space.
        models (List[`str`], *optional*)
            List of models related to this Space. Should be a dataset ID found on https://hf.co/models.
        datasets (`List[str]`, *optional*)
            List of datasets related to this Space. Should be a dataset ID found on https://hf.co/datasets.
        tags (`List[str]`, *optional*)
            List of tags to add to your Space that can be used when filtering on the Hub.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
        kwargs (`dict`, *optional*):
            Additional metadata that will be added to the space card.

    Example:
        ```python
        >>> from huggingface_hub import SpaceCardData
        >>> card_data = SpaceCardData(
        ...     title="Dreambooth Training",
        ...     license="mit",
        ...     sdk="gradio",
        ...     duplicated_from="multimodalart/dreambooth-training"
        ... )
        >>> card_data.to_dict()
        {'title': 'Dreambooth Training', 'sdk': 'gradio', 'license': 'mit', 'duplicated_from': 'multimodalart/dreambooth-training'}
        ```
    """

    def __init__(self, *, title: Optional[str]=None, sdk: Optional[str]=None, sdk_version: Optional[str]=None, python_version: Optional[str]=None, app_file: Optional[str]=None, app_port: Optional[int]=None, license: Optional[str]=None, duplicated_from: Optional[str]=None, models: Optional[List[str]]=None, datasets: Optional[List[str]]=None, tags: Optional[List[str]]=None, ignore_metadata_errors: bool=False, **kwargs):
        self.title = title
        self.sdk = sdk
        self.sdk_version = sdk_version
        self.python_version = python_version
        self.app_file = app_file
        self.app_port = app_port
        self.license = license
        self.duplicated_from = duplicated_from
        self.models = models
        self.datasets = datasets
        self.tags = tags
        super().__init__(**kwargs)