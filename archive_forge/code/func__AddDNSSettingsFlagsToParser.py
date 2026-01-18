from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddDNSSettingsFlagsToParser(parser, mutation_op):
    """Get flags for providing DNS settings.

  Args:
    parser: argparse parser to which to add these flags.
    mutation_op: operation for which we're adding flags.
  """
    dnssec_help_text = ''
    group_help_text = '      Set the authoritative name servers for the given domain.\n      '
    if mutation_op == MutationOp.REGISTER or mutation_op == MutationOp.UPDATE:
        dnssec_help_text = 'If the zone is signed, DNSSEC will be enabled by default unless you pass --disable-dnssec.'
    if mutation_op == MutationOp.UPDATE:
        group_help_text = group_help_text + '\n\n    Warning: Do not change name servers if ds_records is non-empty.\n    Clear ds_records first by calling this command with the\n    --disable-dnssec flag, and wait 24 hours before changing\n    name servers. Otherwise your domain may stop serving.\n\n        '
    if mutation_op == MutationOp.TRANSFER:
        dnssec_help_text = 'DNSSEC will be disabled and will need to be enabled after the transfer completes, if desired.'
        group_help_text = group_help_text + '\n\n    Warning: If your DNS is hosted by your old registrar, we do not\n    recommend keeping your current DNS settings, as these services\n    often terminate when you transfer out. Instead, you should\n    switch to another DNS provider such as Cloud DNS. To avoid\n    downtime during the transfer, copy your DNS records to your new\n    DNS provider before initiating transfer.\n\n    Warning: If you are changing your DNS settings and your domain\n    currently has DS records, make sure to remove the DS records at\n    your old registrar and wait a day before initiating transfer.\n    If you are keeping your current DNS settings, then no changes\n    to DS records are necessary.\n\n        '
    dns_group = base.ArgumentGroup(mutex=True, help=group_help_text, category=base.COMMONLY_USED_FLAGS)
    if mutation_op != MutationOp.TRANSFER:
        dns_group.AddArgument(base.Argument('--name-servers', help='List of DNS name servers for the domain.', metavar='NAME_SERVER', type=arg_parsers.ArgList(str, min_length=2)))
    cloud_dns_transfer_help_text = ''
    if mutation_op == MutationOp.TRANSFER:
        cloud_dns_transfer_help_text = ' To avoid downtime following transfer, make sure the zone is configured correctly before proceeding.'
    cloud_dns_help_text = "The name of the Cloud DNS managed-zone to set as the name server for the domain.\nIf it's in the same project, you can use short name. If not, use the full resource name, e.g.: --cloud-dns-zone=projects/example-project/managedZones/example-zone.{}\n{}".format(cloud_dns_transfer_help_text, dnssec_help_text)
    google_dns_transfer_help_text = ''
    if mutation_op == MutationOp.TRANSFER:
        google_dns_transfer_help_text = ' This blank-slate option cannot be configured before transfer.'
    google_dns_help_text = 'Use free name servers provided by Google Domains.{}\n{}'.format(google_dns_transfer_help_text, dnssec_help_text)
    dns_group.AddArgument(base.Argument('--cloud-dns-zone', help=cloud_dns_help_text))
    dns_group.AddArgument(base.Argument('--use-google-domains-dns', help=google_dns_help_text, default=False, action=actions.DeprecationAction('--use-google-domains-dns', warn='The {flag_name} option is deprecated; See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.', removed=mutation_op == MutationOp.REGISTER, action='store_true')))
    if mutation_op == MutationOp.TRANSFER:
        dns_group.AddArgument(base.Argument('--keep-dns-settings', help="Keep the domain's current DNS configuration from its current registrar. Use this option only if you are sure that the domain's current DNS service will not cease upon transfer, as is often the case for DNS services provided for free by the registrar.", default=False, action='store_true'))
    if mutation_op == MutationOp.UPDATE:
        help_text = '    A YAML file containing the required DNS settings.\n    If specified, its content will replace the values currently used in the\n    registration resource. If the file is missing some of the dns_settings\n    fields, those fields will be cleared.\n\n    Examples of file contents:\n\n    ```\n    googleDomainsDns:\n      dsState: DS_RECORDS_PUBLISHED\n    glueRecords:\n    - hostName: ns1.example.com\n      ipv4Addresses:\n      - 8.8.8.8\n    - hostName: ns2.example.com\n      ipv4Addresses:\n      - 8.8.8.8\n    ```\n\n    ```\n    customDns:\n      nameServers:\n      - new.ns1.com\n      - new.ns2.com\n      dsRecords:\n      - keyTag: 24\n        algorithm: RSASHA1\n        digestType: SHA256\n        digest: 2e1cfa82b035c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d\n      - keyTag: 42\n        algorithm: RSASHA1\n        digestType: SHA256\n        digest: 2e1cfa82bf35c26cbbbdae632cea070514eb8b773f616aaeaf668e2f0be8f10d\n    ```\n        '
        dns_group.AddArgument(base.Argument('--dns-settings-from-file', help=help_text, metavar='DNS_SETTINGS_FILE_NAME'))
    dns_group.AddToParser(parser)
    if mutation_op != MutationOp.TRANSFER:
        base.Argument('--disable-dnssec', help='        Use this flag to disable DNSSEC, or to skip enabling it when switching\n        to a Cloud DNS Zone or Google Domains nameservers.\n        ', default=False, action='store_true').AddToParser(parser)