import datetime
import sys
import time
import urllib
import platform
from typing import Any, Callable, cast, Dict, List, Optional
import json.decoder as jd
import requests
import cirq_ionq
from cirq_ionq import ionq_exceptions
from cirq import __version__ as cirq_version
def _is_retriable(code, method):
    return code in RETRIABLE_STATUS_CODES or (method == 'GET' and code in RETRIABLE_FOR_GETS)