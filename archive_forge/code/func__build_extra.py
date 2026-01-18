import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_extra(self):
    """User agent string components based on legacy "extra" settings.

        Creates components from the session-level and client-level
        ``user_agent_extra`` setting, if present. Both are passed through
        verbatim and should be appended at the end of the string.

        Preferred ways to inject application-specific information into
        botocore's User-Agent header string are the ``user_agent_appid` field
        in :py:class:`botocore.config.Config`. The ``AWS_SDK_UA_APP_ID``
        environment variable and the ``sdk_ua_app_id`` configuration file
        setting are alternative ways to set the ``user_agent_appid`` config.
        """
    extra = []
    if self._session_user_agent_extra:
        extra.append(RawStringUserAgentComponent(self._session_user_agent_extra))
    if self._client_config and self._client_config.user_agent_extra:
        extra.append(RawStringUserAgentComponent(self._client_config.user_agent_extra))
    return extra