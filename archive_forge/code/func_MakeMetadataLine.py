from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import sys
import six
from gslib.cloud_api import EncryptionException
from gslib.exception import CommandException
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GenerationFromUrlAndString
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import text_util
from gslib.wildcard_iterator import StorageUrlFromString
def MakeMetadataLine(label, value, indent=1):
    """Returns a string with a vertically aligned label and value.

  Labels of the same indentation level will start at the same column. Values
  will all start at the same column (unless the combined left-indent and
  label length is excessively long). If a value spans multiple lines,
  indentation will only be applied to the first line. Example output from
  several calls:

      Label1:            Value (default indent of 1 was used)
          Sublabel1:     Value (used indent of 2 here)
      Label2:            Value

  Args:
    label: The label to print in the first column.
    value: The value to print in the second column.
    indent: (4 * indent) spaces will be placed before the label.
  Returns:
    A string with a vertically aligned label and value.
  """
    return '{}{}'.format((' ' * indent * 4 + label + ':').ljust(28), value)