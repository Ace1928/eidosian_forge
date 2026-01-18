from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import route_table
def _vpc_route_tables(self, ignore_errors=False):
    for res in self.stack.values():
        if res.has_interface('AWS::EC2::RouteTable'):
            try:
                vpc_id = self.properties[self.VPC_ID]
                rt_vpc_id = res.properties.get(route_table.RouteTable.VPC_ID)
            except (ValueError, TypeError):
                if ignore_errors:
                    continue
                else:
                    raise
            if rt_vpc_id == vpc_id:
                yield res