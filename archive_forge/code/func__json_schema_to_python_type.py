from __future__ import annotations
import asyncio
import base64
import copy
import json
import mimetypes
import os
import pkgutil
import secrets
import shutil
import tempfile
import warnings
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Optional, TypedDict
import fsspec.asyn
import httpx
import huggingface_hub
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if schema == {}:
        return 'Any'
    type_ = get_type(schema)
    if type_ == {}:
        if 'json' in schema.get('description', {}):
            return 'Dict[Any, Any]'
        else:
            return 'Any'
    elif type_ == '$ref':
        return _json_schema_to_python_type(defs[schema['$ref'].split('/')[-1]], defs)
    elif type_ == 'null':
        return 'None'
    elif type_ == 'const':
        return f'Literal[{schema['const']}]'
    elif type_ == 'enum':
        return 'Literal[' + ', '.join(["'" + str(v) + "'" for v in schema['enum']]) + ']'
    elif type_ == 'integer':
        return 'int'
    elif type_ == 'string':
        return 'str'
    elif type_ == 'boolean':
        return 'bool'
    elif type_ == 'number':
        return 'float'
    elif type_ == 'array':
        items = schema.get('items', [])
        if 'prefixItems' in items:
            elements = ', '.join([_json_schema_to_python_type(i, defs) for i in items['prefixItems']])
            return f'Tuple[{elements}]'
        elif 'prefixItems' in schema:
            elements = ', '.join([_json_schema_to_python_type(i, defs) for i in schema['prefixItems']])
            return f'Tuple[{elements}]'
        else:
            elements = _json_schema_to_python_type(items, defs)
            return f'List[{elements}]'
    elif type_ == 'object':

        def get_desc(v):
            return f' ({v.get('description')})' if v.get('description') else ''
        props = schema.get('properties', {})
        des = [f'{n}: {_json_schema_to_python_type(v, defs)}{get_desc(v)}' for n, v in props.items() if n != '$defs']
        if 'additionalProperties' in schema:
            des += [f'str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}']
        des = ', '.join(des)
        return f'Dict({des})'
    elif type_ in ['oneOf', 'anyOf']:
        desc = ' | '.join([_json_schema_to_python_type(i, defs) for i in schema[type_]])
        return desc
    elif type_ == 'allOf':
        data = ', '.join((_json_schema_to_python_type(i, defs) for i in schema[type_]))
        desc = f'All[{data}]'
        return desc
    else:
        raise APIInfoParseError(f'Cannot parse schema {schema}')