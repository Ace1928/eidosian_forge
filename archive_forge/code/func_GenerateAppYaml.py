from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def GenerateAppYaml(self, notify):
    """Generate app.yaml.

    Args:
      notify: depending on whether we're in deploy, write messages to the
        user or to log.
    Returns:
      (bool) True if file was written

    Note: this is not a recommended use-case,
    python-compat users likely have an existing app.yaml.  But users can
    still get here with the --runtime flag.
    """
    if not self.params.appinfo:
        app_yaml = os.path.join(self.root, 'app.yaml')
        if not os.path.exists(app_yaml):
            notify('Writing [app.yaml] to [%s].' % self.root)
            runtime = 'custom' if self.params.custom else self.runtime
            files.WriteFileContents(app_yaml, PYTHON_APP_YAML.format(runtime=runtime))
            log.warning(APP_YAML_WARNING)
            return True
    return False