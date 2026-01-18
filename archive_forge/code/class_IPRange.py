from boto.ec2.securitygroup import SecurityGroup
class IPRange(object):
    """
    Describes a CIDR address range for use in a DBSecurityGroup

    :ivar cidr_ip: IP Address range
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.cidr_ip = None
        self.status = None

    def __repr__(self):
        return 'IPRange:%s' % self.cidr_ip

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'CIDRIP':
            self.cidr_ip = value
        elif name == 'Status':
            self.status = value
        else:
            setattr(self, name, value)