import json
import re
import typing as ty
from requests import exceptions as _rex
class ForbiddenException(HttpException):
    """HTTP 403 Forbidden Request."""