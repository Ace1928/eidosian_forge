import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
def _load_from_package(filename):
    filepath = os.path.join(path, filename)
    return (filepath, filename, resource_stream(name, filepath), None)