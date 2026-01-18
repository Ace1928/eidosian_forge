from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def _write_records_to_file(self, records):
    bucket_size = 1000000
    bucket = []
    encoder = RecordsEncoder(record_type=self.type, separator=self.separator)
    for record in records:
        result = encoder.encode(record)
        if result:
            bucket.append(to_text(result + ',\n'))
            if len(bucket) == bucket_size:
                self._values['records_src'].writelines(bucket)
                bucket = []
    self._values['records_src'].writelines(bucket)
    self._values['records_src'].seek(0)