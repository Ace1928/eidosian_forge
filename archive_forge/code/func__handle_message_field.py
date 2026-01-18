from __future__ import annotations
import asyncio
import copy
import json
import logging
import typing as t
import warnings
from datetime import datetime, timezone
from jsonschema import ValidationError
from pythonjsonlogger import jsonlogger
from traitlets import Dict, Instance, Set, default
from traitlets.config import Config, LoggingConfigurable
from .schema import SchemaType
from .schema_registry import SchemaRegistry
from .traits import Handlers
from .validators import JUPYTER_EVENTS_CORE_VALIDATOR
def _handle_message_field(record: t.Any, **kwargs: t.Any) -> str:
    """Python's logger always emits the "message" field with
            the value as "null" unless it's present in the schema/data.
            Message happens to be a common field for event logs,
            so special case it here and only emit it if "message"
            is found the in the schema's property list.
            """
    schema = self.schemas.get(record['__schema__'])
    if 'message' not in schema.properties:
        del record['message']
    return json.dumps(record, **kwargs)