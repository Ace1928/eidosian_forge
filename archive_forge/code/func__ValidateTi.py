from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def _ValidateTi(self):
    """Validation specifically for Ti-runtimes."""
    if not self.is_ti_runtime:
        return
    _CheckIllegalAttribute(name='threadsafe', yaml_info=self.parsed, extractor_func=lambda yaml: yaml.threadsafe, file_path=self.file, msg=HINT_THREADSAFE.format(self.runtime))
    for handler in self.parsed.handlers:
        _CheckIllegalAttribute(name='application_readable', yaml_info=handler, extractor_func=lambda yaml: handler.application_readable, file_path=self.file, msg=HINT_READABLE.format(self.runtime))