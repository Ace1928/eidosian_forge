from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryUpdateArgs(parser):
    """Add args for updating an Active Directory."""
    concept_parsers.ConceptParser([flags.GetActiveDirectoryPresentationSpec('The Active Directory to update.')]).AddToParser(parser)
    flags.AddResourceDescriptionArg(parser, 'Active Directory')
    flags.AddResourceAsyncFlag(parser)
    labels_util.AddUpdateLabelsFlags(parser)
    AddActiveDirectoryDomainArg(parser)
    AddActiveDirectorySiteArg(parser)
    AddActiveDirectoryDnsArg(parser)
    AddActiveDirectoryNetBiosArg(parser)
    AddActiveDirectoryOrganizationalUnitArg(parser)
    AddActiveDirectoryAesEncryptionArg(parser)
    AddActiveDirectoryUsernameArg(parser)
    AddActiveDirectoryPasswordArg(parser)
    AddActiveDirectoryBackupOperatorsArg(parser)
    AddActiveDirectorySecurityOperatorsArg(parser)
    AddActivevDirectoryKdcHostnameArg(parser)
    AddActiveDirectoryKdcIpArg(parser)
    AddActiveDirectoryNfsUsersWithLdapArg(parser)
    AddActiveDirectoryLdapSigningArg(parser)
    AddActiveDirectoryEncryptDcConnectionsArg(parser)