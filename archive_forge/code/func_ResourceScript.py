import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
def ResourceScript(path, registry):
    """
    I am a normal py file which must define a 'resource' global, which should
    be an instance of (a subclass of) web.resource.Resource; it will be
    renderred.
    """
    cs = CacheScanner(path, registry)
    glob = {'__file__': _coerceToFilesystemEncoding('', path), 'resource': noRsrc, 'registry': registry, 'cache': cs.cache, 'recache': cs.recache}
    try:
        execfile(path, glob, glob)
    except AlreadyCached as ac:
        return ac.args[0]
    rsrc = glob['resource']
    if cs.doCache and rsrc is not noRsrc:
        registry.cachePath(path, rsrc)
    return rsrc