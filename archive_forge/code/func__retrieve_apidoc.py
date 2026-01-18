from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _retrieve_apidoc(self):
    try:
        os.makedirs(self.apidoc_cache_dir)
    except OSError as err:
        if err.errno != errno.EEXIST or not os.path.isdir(self.apidoc_cache_dir):
            raise
    response = None
    if self.language:
        response = self._retrieve_apidoc_call('/apidoc/v{0}.{1}.json'.format(self.api_version, self.language), safe=True)
        language_family = self.language.split('_')[0]
        if not response and language_family != self.language:
            response = self._retrieve_apidoc_call('/apidoc/v{0}.{1}.json'.format(self.api_version, language_family), safe=True)
    if not response:
        try:
            response = self._retrieve_apidoc_call('/apidoc/v{}.json'.format(self.api_version))
        except Exception as exc:
            raise DocLoadingError('Could not load data from {0}: {1}\n                  - is your server down?'.format(self.uri, exc))
    if not response:
        raise DocLoadingError('Could not load data from {0}'.format(self.uri))
    with open(self.apidoc_cache_file, 'w') as apidoc_file:
        apidoc_file.write(json.dumps(response))
    return response