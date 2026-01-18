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
class ResourceComponent:
    """
    Mix-in class for components that define a set of resources
    that have to be resolved.
    """
    _resources = {'css': {}, 'font': {}, 'js': {}, 'js_modules': {}, 'raw_css': []}

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

    def resolve_resources(self, cdn: bool | Literal['auto']='auto', extras: dict[str, dict[str, str]] | None=None) -> ResourcesType:
        """
        Resolves the resources required for this component.

        Arguments
        ---------
        cdn: bool | Literal['auto']
            Whether to load resources from CDN or local server. If set
            to 'auto' value will be automatically determine based on
            global settings.
        extras: dict[str, dict[str, str]] | None
            Additional resources to add to the bundle. Valid resource
            types include js, js_modules and css.

        Returns
        -------
        Dictionary containing JS and CSS resources.
        """
        cls = type(self)
        resources = {}
        for rt, res in self._resources.items():
            if not isinstance(res, dict):
                continue
            if rt == 'font':
                rt = 'css'
            res = {name: url if isurl(url) else f'{cls.__name__.lower()}/{url}' for name, url in res.items()}
            if rt in resources:
                resources[rt] = dict(resources[rt], **res)
            else:
                resources[rt] = res
        resource_types: ResourcesType = {'js': {}, 'js_modules': {}, 'css': {}, 'raw_css': []}
        cdn = use_cdn() if cdn == 'auto' else cdn
        for resource_type in resource_types:
            if resource_type not in resources or resource_type == 'raw_css':
                continue
            resource_files = resource_types[resource_type]
            for rname, resource in resources[resource_type].items():
                resolved_resource = self._resolve_resource(resource_type, resource, cdn=cdn)
                if resolved_resource:
                    resource_files[rname] = resolved_resource
        version_suffix = f'?v={JS_VERSION}'
        dist_path = get_dist_path(cdn=cdn)
        for resource_type, extra_resources in (extras or {}).items():
            resource_files = resource_types[resource_type]
            for name, res in extra_resources.items():
                if not cdn:
                    res = res.replace(CDN_DIST, dist_path)
                    if not res.endswith(version_suffix):
                        res += version_suffix
                resource_files[name] = res
        return resource_types