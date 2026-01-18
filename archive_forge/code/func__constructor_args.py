from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
def _constructor_args(self):
    args = {'audience': self._audience, 'subject_token_type': self._subject_token_type, 'token_url': self._token_url, 'token_info_url': self._token_info_url, 'service_account_impersonation_url': self._service_account_impersonation_url, 'service_account_impersonation_options': copy.deepcopy(self._service_account_impersonation_options) or None, 'credential_source': copy.deepcopy(self._credential_source), 'quota_project_id': self._quota_project_id, 'client_id': self._client_id, 'client_secret': self._client_secret, 'workforce_pool_user_project': self._workforce_pool_user_project, 'scopes': self._scopes, 'default_scopes': self._default_scopes, 'universe_domain': self._universe_domain}
    if not self.is_workforce_pool:
        args.pop('workforce_pool_user_project')
    return args