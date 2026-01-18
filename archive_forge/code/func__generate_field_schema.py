from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _generate_field_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a schema for the search index in Redis based on the input metadata.

    Given a dictionary of metadata, this function categorizes each metadata
        field into one of the three categories:
    - text: The field contains textual data.
    - numeric: The field contains numeric data (either integer or float).
    - tag: The field contains list of tags (strings).

    Args
        data (Dict[str, Any]): A dictionary where keys are metadata field names
            and values are the metadata values.

    Returns:
        Dict[str, Any]: A dictionary with three keys "text", "numeric", and "tag".
            Each key maps to a list of fields that belong to that category.

    Raises:
        ValueError: If a metadata field cannot be categorized into any of
            the three known types.
    """
    result: Dict[str, Any] = {'text': [], 'numeric': [], 'tag': []}
    for key, value in data.items():
        try:
            int(value)
            result['numeric'].append({'name': key})
            continue
        except (ValueError, TypeError):
            pass
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value or isinstance(value[0], str):
                result['tag'].append({'name': key})
            else:
                name = type(value[0]).__name__
                raise ValueError(f"List/tuple values should contain strings: '{key}': {name}")
            continue
        if isinstance(value, str):
            result['text'].append({'name': key})
            continue
        name = type(value).__name__
        raise ValueError('Could not generate Redis index field type mapping ' + f"for metadata: '{key}': {name}")
    return result