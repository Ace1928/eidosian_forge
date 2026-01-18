from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import re
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64

        Given a Criterium object, creates a ChainMatcher object.
        