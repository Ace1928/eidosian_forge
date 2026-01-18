import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union
from ..repocard_data import ModelCardData

    A class that converts human-readable model search parameters into ones
    compatible with the REST API. For all parameters capitalization does not
    matter.

    <Tip warning={true}>

        The `ModelFilter` class is deprecated and will be removed in huggingface_hub>=0.24. Please pass the filter parameters as keyword arguments directly to [`list_models`].

    </Tip>

    Args:
        author (`str`, *optional*):
            A string that can be used to identify models on the Hub by the
            original uploader (author or organization), such as `facebook` or
            `huggingface`.
        library (`str` or `List`, *optional*):
            A string or list of strings of foundational libraries models were
            originally trained from, such as pytorch, tensorflow, or allennlp.
        language (`str` or `List`, *optional*):
            A string or list of strings of languages, both by name and country
            code, such as "en" or "English"
        model_name (`str`, *optional*):
            A string that contain complete or partial names for models on the
            Hub, such as "bert" or "bert-base-cased"
        task (`str` or `List`, *optional*):
            A string or list of strings of tasks models were designed for, such
            as: "fill-mask" or "automatic-speech-recognition"
        tags (`str` or `List`, *optional*):
            A string tag or a list of tags to filter models on the Hub by, such
            as `text-generation` or `spacy`.
        trained_dataset (`str` or `List`, *optional*):
            A string tag or a list of string tags of the trained dataset for a
            model on the Hub.

    Examples:

    ```python
    >>> from huggingface_hub import ModelFilter

    >>> # For the author_or_organization
    >>> new_filter = ModelFilter(author_or_organization="facebook")

    >>> # For the library
    >>> new_filter = ModelFilter(library="pytorch")

    >>> # For the language
    >>> new_filter = ModelFilter(language="french")

    >>> # For the model_name
    >>> new_filter = ModelFilter(model_name="bert")

    >>> # For the task
    >>> new_filter = ModelFilter(task="text-classification")

    >>> from huggingface_hub import HfApi

    >>> api = HfApi()
    # To list model tags

    >>> new_filter = ModelFilter(tags="benchmark:raft")

    >>> # Related to the dataset
    >>> new_filter = ModelFilter(trained_dataset="common_voice")
    ```
    