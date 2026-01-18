from datetime import datetime
from boto.ec2.cloudwatch.listelement import ListElement
from boto.ec2.cloudwatch.dimension import Dimension
from boto.compat import json
from boto.compat import six
def add_alarm_action(self, action_arn=None):
    """
        Adds an alarm action, represented as an SNS topic, to this alarm.
        What do do when alarm is triggered.

        :type action_arn: str
        :param action_arn: SNS topics to which notification should be
                           sent if the alarm goes to state ALARM.
        """
    if not action_arn:
        return
    self.actions_enabled = 'true'
    self.alarm_actions.append(action_arn)