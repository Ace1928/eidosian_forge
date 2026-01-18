from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
Uploads a file-like object via the REST API to a given filename

        Args:
            content: The file-like object whose content to upload
            name: The remote name of the file to store the content in. The
                  final location of the file will be in /var/config/rest/downloads.

        Returns:
            void
        