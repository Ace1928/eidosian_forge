from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from gslib.cloud_api import ArgumentException
from gslib.utils.text_util import AddQueryParamToUrl
Ensures dst_obj_metadata supplies the needed fields for copy and insert.

  Args:
    dst_obj_metadata: Metadata to validate.

  Raises:
    ArgumentException if metadata is invalid.
  