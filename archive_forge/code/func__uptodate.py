import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
def _uptodate():
    return mtime == os.path.getmtime(filepath)