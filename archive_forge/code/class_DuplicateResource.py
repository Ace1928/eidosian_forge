import json
import re
import typing as ty
from requests import exceptions as _rex
class DuplicateResource(SDKException):
    """More than one resource exists with that name."""