from boto.ec2.ec2object import TaggedEC2Object
class VpcPeeringConnectionStatus(object):
    """
    The status of VPC peering connection.

    :ivar code: The status of the VPC peering connection. Valid values are:

        * pending-acceptance
        * failed
        * expired
        * provisioning
        * active
        * deleted
        * rejected

    :ivar message: A message that provides more information about the status of the VPC peering connection, if applicable.
    """

    def __init__(self, code=0, message=None):
        self.code = code
        self.message = message

    def __repr__(self):
        return '%s(%d)' % (self.code, self.message)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'code':
            self.code = value
        elif name == 'message':
            self.message = value
        else:
            setattr(self, name, value)