from __future__ import annotations
import copy
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import requests
import yaml
from langchain_core.pydantic_v1 import ValidationError
@staticmethod
def _alert_unsupported_spec(obj: dict) -> None:
    """Alert if the spec is not supported."""
    warning_message = ' This may result in degraded performance.' + ' Convert your OpenAPI spec to 3.1.* spec' + ' for better support.'
    swagger_version = obj.get('swagger')
    openapi_version = obj.get('openapi')
    if isinstance(openapi_version, str):
        if openapi_version != '3.1.0':
            logger.warning(f'Attempting to load an OpenAPI {openapi_version} spec. {warning_message}')
        else:
            pass
    elif isinstance(swagger_version, str):
        logger.warning(f'Attempting to load a Swagger {swagger_version} spec. {warning_message}')
    else:
        raise ValueError(f'Attempting to load an unsupported spec:\n\n{obj}\n{warning_message}')