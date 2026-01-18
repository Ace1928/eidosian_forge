from boto.resultset import ResultSet
from boto.ec2.elb.listelement import ListElement
class ScalingPolicy(object):

    def __init__(self, connection=None, **kwargs):
        """
        Scaling Policy

        :type name: str
        :param name: Name of scaling policy.

        :type adjustment_type: str
        :param adjustment_type: Specifies the type of adjustment. Valid values are `ChangeInCapacity`, `ExactCapacity` and `PercentChangeInCapacity`.

        :type as_name: str or int
        :param as_name: Name or ARN of the Auto Scaling Group.

        :type scaling_adjustment: int
        :param scaling_adjustment: Value of adjustment (type specified in `adjustment_type`).

        :type min_adjustment_step: int
        :param min_adjustment_step: Value of min adjustment step required to
            apply the scaling policy (only make sense when use `PercentChangeInCapacity` as adjustment_type.).

        :type cooldown: int
        :param cooldown: Time (in seconds) before Alarm related Scaling Activities can start after the previous Scaling Activity ends.

        """
        self.name = kwargs.get('name', None)
        self.adjustment_type = kwargs.get('adjustment_type', None)
        self.as_name = kwargs.get('as_name', None)
        self.scaling_adjustment = kwargs.get('scaling_adjustment', None)
        self.cooldown = kwargs.get('cooldown', None)
        self.connection = connection
        self.min_adjustment_step = kwargs.get('min_adjustment_step', None)

    def __repr__(self):
        return 'ScalingPolicy(%s group:%s adjustment:%s)' % (self.name, self.as_name, self.adjustment_type)

    def startElement(self, name, attrs, connection):
        if name == 'Alarms':
            self.alarms = ResultSet([('member', Alarm)])
            return self.alarms

    def endElement(self, name, value, connection):
        if name == 'PolicyName':
            self.name = value
        elif name == 'AutoScalingGroupName':
            self.as_name = value
        elif name == 'PolicyARN':
            self.policy_arn = value
        elif name == 'ScalingAdjustment':
            self.scaling_adjustment = int(value)
        elif name == 'Cooldown':
            self.cooldown = int(value)
        elif name == 'AdjustmentType':
            self.adjustment_type = value
        elif name == 'MinAdjustmentStep':
            self.min_adjustment_step = int(value)

    def delete(self):
        return self.connection.delete_policy(self.name, self.as_name)