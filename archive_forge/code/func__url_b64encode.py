import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def _url_b64encode(self, data):
    return base64.b64encode(data).replace(b'+', b'-').replace(b'=', b'_').replace(b'/', b'~')