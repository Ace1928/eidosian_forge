import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
class Css:

    def __init__(self, serve_locally):
        self._resources = Resources('_css_dist')
        self._resources.config = self.config = _Config(serve_locally, True)

    def append_css(self, stylesheet):
        self._resources.append_resource(stylesheet)

    def get_all_css(self):
        return self._resources.get_all_resources()

    def get_library_css(self, libraries):
        return self._resources.get_library_resources(libraries)