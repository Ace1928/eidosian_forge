from boto.ec2.ec2object import TaggedEC2Object
class VpcPeeringConnection(TaggedEC2Object):

    def __init__(self, connection=None):
        """
        Represents a VPC peering connection.

        :ivar id: The unique ID of the VPC peering connection.
        :ivar accepter_vpc_info: Information on peer Vpc.
        :ivar requester_vpc_info: Information on requester Vpc.
        :ivar expiration_time: The expiration date and time for the VPC peering connection.
        :ivar status_code: The status of the VPC peering connection.
        :ivar status_message: A message that provides more information about the status of the VPC peering connection, if applicable.
        """
        super(VpcPeeringConnection, self).__init__(connection)
        self.id = None
        self.accepter_vpc_info = VpcInfo()
        self.requester_vpc_info = VpcInfo()
        self.expiration_time = None
        self._status = VpcPeeringConnectionStatus()

    @property
    def status_code(self):
        return self._status.code

    @property
    def status_message(self):
        return self._status.message

    def __repr__(self):
        return 'VpcPeeringConnection:%s' % self.id

    def startElement(self, name, attrs, connection):
        retval = super(VpcPeeringConnection, self).startElement(name, attrs, connection)
        if retval is not None:
            return retval
        if name == 'requesterVpcInfo':
            return self.requester_vpc_info
        elif name == 'accepterVpcInfo':
            return self.accepter_vpc_info
        elif name == 'status':
            return self._status
        return None

    def endElement(self, name, value, connection):
        if name == 'vpcPeeringConnectionId':
            self.id = value
        elif name == 'expirationTime':
            self.expiration_time = value
        else:
            setattr(self, name, value)

    def delete(self):
        return self.connection.delete_vpc_peering_connection(self.id)

    def _update(self, updated):
        self.__dict__.update(updated.__dict__)

    def update(self, validate=False, dry_run=False):
        vpc_peering_connection_list = self.connection.get_all_vpc_peering_connections([self.id], dry_run=dry_run)
        if len(vpc_peering_connection_list):
            updated_vpc_peering_connection = vpc_peering_connection_list[0]
            self._update(updated_vpc_peering_connection)
        elif validate:
            raise ValueError('%s is not a valid VpcPeeringConnection ID' % (self.id,))
        return self.status_code