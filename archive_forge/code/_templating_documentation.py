from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union
from ..utils import HfHubHTTPError, RepositoryNotFoundError, is_minijinja_available
Fetch and compile a model's chat template.

    Method is cached to avoid fetching the same model's config multiple times.

    Args:
        model_id (`str`):
            The model id.
        token (`str` or `bool`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.

    Returns:
        `Callable`: A callable that takes a list of messages and returns the rendered chat prompt.
    