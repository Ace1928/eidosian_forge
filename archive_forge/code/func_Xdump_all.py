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
def Xdump_all(self, documents, stream, _kw=enforce, transform=None):
    """
        Serialize a sequence of Python objects into a YAML stream.
        """
    if not hasattr(stream, 'write') and hasattr(stream, 'open'):
        with stream.open('w') as fp:
            return self.dump_all(documents, fp, _kw, transform=transform)
    if _kw is not enforce:
        raise TypeError('{}.dump(_all) takes two positional argument but at least three were given ({!r})'.format(self.__class__.__name__, _kw))
    if self.top_level_colon_align is True:
        tlca = max([len(str(x)) for x in documents[0]])
    else:
        tlca = self.top_level_colon_align
    if transform is not None:
        fstream = stream
        if self.encoding is None:
            stream = StringIO()
        else:
            stream = BytesIO()
    serializer, representer, emitter = self.get_serializer_representer_emitter(stream, tlca)
    try:
        self.serializer.open()
        for data in documents:
            try:
                self.representer.represent(data)
            except AttributeError:
                raise
        self.serializer.close()
    finally:
        try:
            self.emitter.dispose()
        except AttributeError:
            raise
        delattr(self, '_serializer')
        delattr(self, '_emitter')
    if transform:
        val = stream.getvalue()
        if self.encoding:
            val = val.decode(self.encoding)
        if fstream is None:
            transform(val)
        else:
            fstream.write(transform(val))
    return None