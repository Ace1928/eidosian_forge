from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def ChooseImportOrRestoreMethod(unused_instance_ref, args):
    if args.replace_all:
        return 'restore'
    return 'import'