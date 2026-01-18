import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class BadCredentialsException(GithubException):
    """
    Exception raised in case of bad credentials (when Github API replies with a 401 or 403 HTML status)
    """