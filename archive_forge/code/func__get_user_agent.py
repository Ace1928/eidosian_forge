import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, Union
from urllib.parse import urlsplit
import requests
from urllib3.exceptions import InsecureRequestWarning
import sphinx
from sphinx.config import Config
def _get_user_agent(config: Config) -> str:
    if config.user_agent:
        return config.user_agent
    else:
        return ' '.join(['Sphinx/%s' % sphinx.__version__, 'requests/%s' % requests.__version__, 'python/%s' % '.'.join(map(str, sys.version_info[:3]))])