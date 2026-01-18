from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _CheckEnvironment(path):
    """Gathers information about the local environment, and performs some checks.

  Args:
    path: (str) Application path.

  Returns:
    (bool) Whether bundler is available in the environment.

  Raises:
    RubyConfigError: The application is recognized as a Ruby app but
    malformed in some way.
  """
    if not os.path.isfile(os.path.join(path, 'Gemfile')):
        raise MissingGemfileError('Gemfile is required for Ruby runtime.')
    gemfile_lock_present = os.path.isfile(os.path.join(path, 'Gemfile.lock'))
    bundler_available = _SubprocessSucceeds('bundle version')
    if bundler_available:
        if not _SubprocessSucceeds('bundle check'):
            raise StaleBundleError("Your bundle is not up-to-date. Install missing gems with 'bundle install'.")
        if not gemfile_lock_present:
            msg = '\nNOTICE: We could not find a Gemfile.lock, which suggests this application has not been tested locally, or the Gemfile.lock has not been committed to source control. We have created a Gemfile.lock for you, but it is recommended that you verify it yourself (by installing your bundle and testing locally) to ensure that the gems we deploy are the same as those you tested.'
            log.status.Print(msg)
    else:
        msg = "\nNOTICE: gcloud could not run bundler in your local environment, and so its ability to determine your application's requirements will be limited. We will still attempt to deploy your application, but if your application has trouble starting up due to missing requirements, we recommend installing bundler by running [gem install bundler]"
        log.status.Print(msg)
    return bundler_available