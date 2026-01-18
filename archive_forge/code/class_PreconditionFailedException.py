import json
import re
import typing as ty
from requests import exceptions as _rex
class PreconditionFailedException(HttpException):
    """HTTP 412 Precondition Failed."""