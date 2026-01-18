from __future__ import annotations
import base64
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, cast
from urllib.parse import urlparse
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.vertexai import (
from langchain_community.utilities.vertexai import (
def _parse_examples(examples: List[BaseMessage]) -> List['InputOutputTextPair']:
    from vertexai.language_models import InputOutputTextPair
    if len(examples) % 2 != 0:
        raise ValueError(f'Expect examples to have an even amount of messages, got {len(examples)}.')
    example_pairs = []
    input_text = None
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if not isinstance(example, HumanMessage):
                raise ValueError(f'Expected the first message in a part to be from human, got {type(example)} for the {i}th message.')
            input_text = example.content
        if i % 2 == 1:
            if not isinstance(example, AIMessage):
                raise ValueError(f'Expected the second message in a part to be from AI, got {type(example)} for the {i}th message.')
            pair = InputOutputTextPair(input_text=input_text, output_text=example.content)
            example_pairs.append(pair)
    return example_pairs