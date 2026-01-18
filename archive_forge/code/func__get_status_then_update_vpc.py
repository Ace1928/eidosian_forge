from boto.ec2.ec2object import TaggedEC2Object
def _get_status_then_update_vpc(self, get_status_method, validate=False, dry_run=False):
    vpc_list = get_status_method([self.id], dry_run=dry_run)
    if len(vpc_list):
        updated_vpc = vpc_list[0]
        self._update(updated_vpc)
    elif validate:
        raise ValueError('%s is not a valid VPC ID' % (self.id,))