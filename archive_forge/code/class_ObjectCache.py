import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
class ObjectCache(FileCache):
    """
    Pickled object file cache.

    @cvar protocol: The pickling protocol.
    @type protocol: int

    """
    protocol = 2

    def fnsuffix(self):
        return 'px'

    def get(self, id):
        fp = None
        try:
            fp = self._getf(id)
            if fp is not None:
                cached = pickle.load(fp)
                fp.close()
                return cached
        except Exception:
            if fp is not None:
                fp.close()
            self.purge(id)

    def put(self, id, object):
        data = pickle.dumps(object, self.protocol)
        super(ObjectCache, self).put(id, data)
        return object