import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
class ActivityType(SWFBase):
    """A versioned activity type."""
    version = None

    @wraps(Layer1.deprecate_activity_type)
    def deprecate(self):
        """DeprecateActivityType."""
        return self._swf.deprecate_activity_type(self.domain, self.name, self.version)

    @wraps(Layer1.describe_activity_type)
    def describe(self):
        """DescribeActivityType."""
        return self._swf.describe_activity_type(self.domain, self.name, self.version)

    @wraps(Layer1.register_activity_type)
    def register(self, **kwargs):
        """RegisterActivityType."""
        args = {'default_task_heartbeat_timeout': '600', 'default_task_schedule_to_close_timeout': '3900', 'default_task_schedule_to_start_timeout': '300', 'default_task_start_to_close_timeout': '3600'}
        args.update(kwargs)
        self._swf.register_activity_type(self.domain, self.name, self.version, **args)