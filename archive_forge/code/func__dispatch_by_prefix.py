import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
def _dispatch_by_prefix(filename):
    for prefix, delegate in delegates.items():
        if filename.startswith(prefix):
            if isinstance(delegate, six.string_types):
                delegate = directory(delegate)
            filepath, _, fileobj, uptodate = delegate(filename[len(prefix):].lstrip('/\\'))
            return (filepath, filename, fileobj, uptodate)
    raise TemplateNotFound(filename, list(delegates.keys()))