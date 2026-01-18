import contextlib
import copy
import os
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_serialization import jsonutils
import requests
Check ``https:`` rules by calling to a remote server.

    This example implementation simply verifies that the response
    is exactly ``True``.
    