
    Represents an RDS database subnet group

    Properties reference available from the AWS documentation at http://docs.amazonwebservices.com/AmazonRDS/latest/APIReference/API_DeleteDBSubnetGroup.html

    :ivar status: The current status of the subnet group. Possibile values are [ active, ? ]. Reference documentation lacks specifics of possibilities
    :ivar connection: boto.rds.RDSConnection associated with the current object
    :ivar description: The description of the subnet group
    :ivar subnet_ids: List of subnet identifiers in the group
    :ivar name: Name of the subnet group
    :ivar vpc_id: The ID of the VPC the subnets are inside
    