from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
Returns back the JWT header parameters as a dict()

        Note: The signature is not verified so the header parameters
        should not be fully trusted until signature verification is complete
        