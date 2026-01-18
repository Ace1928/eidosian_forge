from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
@classmethod
def _resolve_resource(cls, resource_type: str, resource: str, cdn: bool=False):
    dist_path = get_dist_path(cdn=cdn)
    if resource.startswith(CDN_DIST):
        resource_path = resource.replace(f'{CDN_DIST}bundled/', '')
    elif resource.startswith(config.npm_cdn):
        resource_path = resource.replace(config.npm_cdn, '')[1:]
    elif resource.startswith('http:'):
        resource_path = url_path(resource)
    else:
        resource_path = resource
    if resource_type == 'js_modules' and (not (state.rel_path or cdn)):
        prefixed_dist = f'./{dist_path}'
    else:
        prefixed_dist = dist_path
    bundlepath = BUNDLE_DIR / resource_path.replace('/', os.path.sep)
    try:
        is_file = bundlepath.is_file()
    except Exception:
        is_file = False
    if is_file:
        return f'{prefixed_dist}bundled/{resource_path}'
    elif isurl(resource):
        return resource
    elif resolve_custom_path(cls, resource):
        return component_resource_path(cls, f'_resources/{resource_type}', resource)