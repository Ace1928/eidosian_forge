from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, PY3, nprint
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader
class YAMLContextManager(object):

    def __init__(self, yaml, transform=None):
        self._yaml = yaml
        self._output_inited = False
        self._output_path = None
        self._output = self._yaml._output
        self._transform = transform
        if not hasattr(self._output, 'write') and hasattr(self._output, 'open'):
            self._output_path = self._output
            self._output = self._output_path.open('w')
        if self._transform is not None:
            self._fstream = self._output
            if self._yaml.encoding is None:
                self._output = StringIO()
            else:
                self._output = BytesIO()

    def teardown_output(self):
        if self._output_inited:
            self._yaml.serializer.close()
        else:
            return
        try:
            self._yaml.emitter.dispose()
        except AttributeError:
            raise
        try:
            delattr(self._yaml, '_serializer')
            delattr(self._yaml, '_emitter')
        except AttributeError:
            raise
        if self._transform:
            val = self._output.getvalue()
            if self._yaml.encoding:
                val = val.decode(self._yaml.encoding)
            if self._fstream is None:
                self._transform(val)
            else:
                self._fstream.write(self._transform(val))
                self._fstream.flush()
                self._output = self._fstream
        if self._output_path is not None:
            self._output.close()

    def init_output(self, first_data):
        if self._yaml.top_level_colon_align is True:
            tlca = max([len(str(x)) for x in first_data])
        else:
            tlca = self._yaml.top_level_colon_align
        self._yaml.get_serializer_representer_emitter(self._output, tlca)
        self._yaml.serializer.open()
        self._output_inited = True

    def dump(self, data):
        if not self._output_inited:
            self.init_output(data)
        try:
            self._yaml.representer.represent(data)
        except AttributeError:
            raise