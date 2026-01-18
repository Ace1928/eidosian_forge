import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _inject_env_variables(self, env):
    env['GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE'] = self._audience
    env['GOOGLE_EXTERNAL_ACCOUNT_TOKEN_TYPE'] = self._subject_token_type
    env['GOOGLE_EXTERNAL_ACCOUNT_ID'] = self.external_account_id
    env['GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE'] = '1' if self.interactive else '0'
    if self._service_account_impersonation_url is not None:
        env['GOOGLE_EXTERNAL_ACCOUNT_IMPERSONATED_EMAIL'] = self.service_account_email
    if self._credential_source_executable_output_file is not None:
        env['GOOGLE_EXTERNAL_ACCOUNT_OUTPUT_FILE'] = self._credential_source_executable_output_file