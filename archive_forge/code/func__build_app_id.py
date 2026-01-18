import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_app_id(self):
    """
        Build app component of the User-Agent header string.

        Returns a single component with prefix "app" and value sourced from the
        ``user_agent_appid`` field in :py:class:`botocore.config.Config` or
        the ``sdk_ua_app_id`` setting in the shared configuration file, or the
        ``AWS_SDK_UA_APP_ID`` environment variable. These are the recommended
        ways for apps built with Botocore to insert their identifer into the
        User-Agent header.
        """
    if self._client_config and self._client_config.user_agent_appid:
        return [UserAgentComponent('app', self._client_config.user_agent_appid)]
    else:
        return []