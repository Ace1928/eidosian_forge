from __future__ import annotations
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from langchain_community.vectorstores.azure_cosmos_db import (
from langchain_core._api.deprecation import deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(f'Could not decode json to list of generations: {generations_json}')