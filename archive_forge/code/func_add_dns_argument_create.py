from neutronclient._i18n import _
def add_dns_argument_create(parser, resource, attribute):
    argument = '--dns-%s' % attribute
    parser.add_argument(argument, help=_('Assign DNS %(attribute)s to the %(resource)s (requires DNS integration extension)') % {'attribute': attribute, 'resource': resource})