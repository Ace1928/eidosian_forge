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
def _ChAcl(self):
    """Parses options and changes ACLs on the specified buckets/objects."""
    self.parse_versions = True
    self.changes = []
    self.continue_on_error = False
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-f':
                self.continue_on_error = True
            elif o == '-g':
                if 'gserviceaccount.com' in a:
                    raise CommandException('Service accounts are considered users, not groups; please use "gsutil acl ch -u" instead of "gsutil acl ch -g"')
                self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.GROUP))
            elif o == '-p':
                self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.PROJECT))
            elif o == '-u':
                self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.USER))
            elif o == '-d':
                self.changes.append(acl_helper.AclDel(a))
            elif o == '-r' or o == '-R':
                self.recursion_requested = True
            else:
                self.RaiseInvalidArgumentException()
    if not self.changes:
        raise CommandException('Please specify at least one access change with the -g, -u, or -d flags')
    if not UrlsAreForSingleProvider(self.args) or StorageUrlFromString(self.args[0]).scheme != 'gs':
        raise CommandException('The "{0}" command can only be used with gs:// URLs'.format(self.command_name))
    self.everything_set_okay = True
    self.ApplyAclFunc(_ApplyAclChangesWrapper, _ApplyExceptionHandler, self.args, object_fields=['acl', 'generation', 'metageneration'])
    if not self.everything_set_okay:
        raise CommandException('ACLs for some objects could not be set.')