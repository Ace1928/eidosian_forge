import json
import re
import typing as ty
from requests import exceptions as _rex
class ServiceDisabledException(ConfigException):
    """This service is disabled for reasons."""