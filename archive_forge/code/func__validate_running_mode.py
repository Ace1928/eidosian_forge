import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _validate_running_mode(self):
    env_allow_executables = os.environ.get('GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES')
    if env_allow_executables != '1':
        raise exceptions.MalformedError("Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run.")
    if self.interactive and (not self._credential_source_executable_output_file):
        raise exceptions.MalformedError('An output_file must be specified in the credential configuration for interactive mode.')
    if self.interactive and (not self._credential_source_executable_interactive_timeout_millis):
        raise exceptions.InvalidOperation('Interactive mode cannot run without an interactive timeout.')
    if self.interactive and (not self.is_workforce_pool):
        raise exceptions.InvalidValue('Interactive mode is only enabled for workforce pool.')