import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
def _load_from_directory(filename):
    filepath = os.path.join(path, filename)
    fileobj = open(filepath, 'rb')
    mtime = os.path.getmtime(filepath)

    def _uptodate():
        return mtime == os.path.getmtime(filepath)
    return (filepath, filename, fileobj, _uptodate)