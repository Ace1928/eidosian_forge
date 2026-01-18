from __future__ import annotations
import json
import os.path as osp
from glob import iglob
from itertools import chain
from logging import Logger
from os.path import join as pjoin
from typing import Any
import json5
from jupyter_core.paths import SYSTEM_CONFIG_PATH, jupyter_config_dir, jupyter_path
from jupyter_server.services.config.manager import ConfigManager, recursive_update
from jupyter_server.utils import url_path_join as ujoin
from traitlets import Bool, HasTraits, List, Unicode, default
class LabConfig(HasTraits):
    """The lab application configuration object."""
    app_name = Unicode('', help='The name of the application.').tag(config=True)
    app_version = Unicode('', help='The version of the application.').tag(config=True)
    app_namespace = Unicode('', help='The namespace of the application.').tag(config=True)
    app_url = Unicode('/lab', help='The url path for the application.').tag(config=True)
    app_settings_dir = Unicode('', help='The application settings directory.').tag(config=True)
    extra_labextensions_path = List(Unicode(), help='Extra paths to look for federated JupyterLab extensions').tag(config=True)
    labextensions_path = List(Unicode(), help='The standard paths to look in for federated JupyterLab extensions').tag(config=True)
    templates_dir = Unicode('', help='The application templates directory.').tag(config=True)
    static_dir = Unicode('', help='The optional location of local static files. If given, a static file handler will be added.').tag(config=True)
    labextensions_url = Unicode('', help='The url for federated JupyterLab extensions').tag(config=True)
    settings_url = Unicode(help='The url path of the settings handler.').tag(config=True)
    user_settings_dir = Unicode('', help='The optional location of the user settings directory.').tag(config=True)
    schemas_dir = Unicode('', help='The optional location of the settings schemas directory. If given, a handler will be added for settings.').tag(config=True)
    workspaces_api_url = Unicode(help='The url path of the workspaces API.').tag(config=True)
    workspaces_dir = Unicode('', help='The optional location of the saved workspaces directory. If given, a handler will be added for workspaces.').tag(config=True)
    listings_url = Unicode(help='The listings url.').tag(config=True)
    themes_url = Unicode(help='The theme url.').tag(config=True)
    licenses_url = Unicode(help='The third-party licenses url.')
    themes_dir = Unicode('', help='The optional location of the themes directory. If given, a handler will be added for themes.').tag(config=True)
    translations_api_url = Unicode(help='The url path of the translations handler.').tag(config=True)
    tree_url = Unicode(help='The url path of the tree handler.').tag(config=True)
    cache_files = Bool(True, help='Whether to cache files on the server. This should be `True` except in dev mode.').tag(config=True)
    notebook_starts_kernel = Bool(True, help='Whether a notebook should start a kernel automatically.').tag(config=True)
    copy_absolute_path = Bool(False, help='Whether getting a relative (False) or absolute (True) path when copying a path.').tag(config=True)

    @default('template_dir')
    def _default_template_dir(self) -> str:
        return DEFAULT_TEMPLATE_PATH

    @default('labextensions_url')
    def _default_labextensions_url(self) -> str:
        return ujoin(self.app_url, 'extensions/')

    @default('labextensions_path')
    def _default_labextensions_path(self) -> list[str]:
        return jupyter_path('labextensions')

    @default('workspaces_url')
    def _default_workspaces_url(self) -> str:
        return ujoin(self.app_url, 'workspaces/')

    @default('workspaces_api_url')
    def _default_workspaces_api_url(self) -> str:
        return ujoin(self.app_url, 'api', 'workspaces/')

    @default('settings_url')
    def _default_settings_url(self) -> str:
        return ujoin(self.app_url, 'api', 'settings/')

    @default('listings_url')
    def _default_listings_url(self) -> str:
        return ujoin(self.app_url, 'api', 'listings/')

    @default('themes_url')
    def _default_themes_url(self) -> str:
        return ujoin(self.app_url, 'api', 'themes/')

    @default('licenses_url')
    def _default_licenses_url(self) -> str:
        return ujoin(self.app_url, 'api', 'licenses/')

    @default('tree_url')
    def _default_tree_url(self) -> str:
        return ujoin(self.app_url, 'tree/')

    @default('translations_api_url')
    def _default_translations_api_url(self) -> str:
        return ujoin(self.app_url, 'api', 'translations/')