from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _load_apidoc(self):
    try:
        with open(self.apidoc_cache_file, 'r') as apidoc_file:
            api_doc = json.load(apidoc_file)
    except (IOError, JSONDecodeError):
        api_doc = self._retrieve_apidoc()
    return api_doc