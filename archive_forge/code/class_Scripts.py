import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
class Scripts:

    def __init__(self, serve_locally, eager):
        self._resources = Resources('_js_dist')
        self._resources.config = self.config = _Config(serve_locally, eager)

    def append_script(self, script):
        self._resources.append_resource(script)

    def get_all_scripts(self, dev_bundles=False):
        return self._resources.get_all_resources(dev_bundles)

    def get_library_scripts(self, libraries, dev_bundles=False):
        return self._resources.get_library_resources(libraries, dev_bundles)