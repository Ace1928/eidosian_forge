from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from gslib.cloud_api import ArgumentException
from gslib.utils.text_util import AddQueryParamToUrl
def ListToGetFields(list_fields=None):
    """Removes 'items/' from the input fields and converts it to a set.

  Args:
    list_fields: Iterable fields usable in ListBuckets/ListObjects calls.

  Returns:
    Set of fields usable in GetBucket/GetObjectMetadata calls (None implies
    all fields should be returned).
  """
    if list_fields:
        get_fields = set()
        for field in list_fields:
            if field in ('kind', 'nextPageToken', 'prefixes'):
                continue
            get_fields.add(re.sub('items/', '', field))
        return get_fields