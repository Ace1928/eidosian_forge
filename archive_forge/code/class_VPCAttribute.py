class VPCAttribute(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.vpc_id = None
        self.enable_dns_hostnames = None
        self.enable_dns_support = None
        self._current_attr = None

    def startElement(self, name, attrs, connection):
        if name in ('enableDnsHostnames', 'enableDnsSupport'):
            self._current_attr = name

    def endElement(self, name, value, connection):
        if name == 'vpcId':
            self.vpc_id = value
        elif name == 'value':
            if value == 'true':
                value = True
            else:
                value = False
            if self._current_attr == 'enableDnsHostnames':
                self.enable_dns_hostnames = value
            elif self._current_attr == 'enableDnsSupport':
                self.enable_dns_support = value