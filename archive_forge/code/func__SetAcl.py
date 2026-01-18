from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import encoding
from gslib import metrics
from gslib import gcs_json_api
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import SetAclExceptionHandler
from gslib.command import SetAclFuncWrapper
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.storage_url import RaiseErrorIfUrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import acl_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _SetAcl(self):
    """Parses options and sets ACLs on the specified buckets/objects."""
    self.continue_on_error = False
    if self.sub_opts:
        for o, unused_a in self.sub_opts:
            if o == '-a':
                self.all_versions = True
            elif o == '-f':
                self.continue_on_error = True
            elif o == '-r' or o == '-R':
                self.recursion_requested = True
            else:
                self.RaiseInvalidArgumentException()
    try:
        self.SetAclCommandHelper(SetAclFuncWrapper, SetAclExceptionHandler)
    except AccessDeniedException as unused_e:
        self._WarnServiceAccounts()
        raise
    if not self.everything_set_okay:
        raise CommandException('ACLs for some objects could not be set.')