import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def _check_url(self, url: Optional[str], *, path: str='api') -> Optional[str]:
    """
        Checks if ``url`` starts with a different base URL from the user-provided base
        URL and warns the user before returning it. If ``keep_base_url`` is set to
        ``True``, instead returns the URL massaged to match the user-provided base URL.
        """
    if not url or url.startswith(self.url):
        return url
    match = re.match(f'(^.*?)/{path}', url)
    if not match:
        return url
    base_url = match.group(1)
    if self.keep_base_url:
        return url.replace(base_url, f'{self._base_url}')
    utils.warn(message=f'The base URL in the server response differs from the user-provided base URL ({self.url} -> {base_url}).\nThis is usually caused by a misconfigured base URL on your side or a misconfigured external_url on the server side, and can lead to broken pagination and unexpected behavior. If this is intentional, use `keep_base_url=True` when initializing the Gitlab instance to keep the user-provided base URL.', category=UserWarning)
    return url