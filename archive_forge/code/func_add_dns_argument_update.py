from neutronclient._i18n import _
def add_dns_argument_update(parser, resource, attribute):
    argument = '--dns-%s' % attribute
    no_argument = '--no-dns-%s' % attribute
    dns_args = parser.add_mutually_exclusive_group()
    dns_args.add_argument(argument, help=_('Assign DNS %(attribute)s to the %(resource)s (requires DNS integration extension.)') % {'attribute': attribute, 'resource': resource})
    dns_args.add_argument(no_argument, action='store_true', help=_('Unassign DNS %(attribute)s from the %(resource)s (requires DNS integration extension.)') % {'attribute': attribute, 'resource': resource})