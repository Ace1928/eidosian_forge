import json
class MetricAction:
    actions = ['inc', 'observe']

    def __init__(self, action, value):
        if action not in self.actions:
            raise UnSupportedMetricActionError('%s action is not supported' % action)
        self.action = action
        self.value = value

    @classmethod
    def validate(cls, metric_action_dict):
        if 'value' not in metric_action_dict:
            raise MetricValidationError("action need 'value' field")
        if 'action' not in metric_action_dict:
            raise MetricValidationError("action need 'action' field")
        if metric_action_dict['action'] not in cls.actions:
            raise MetricValidationError('action should be choosen from %s' % cls.actions)

    @classmethod
    def from_dict(cls, metric_action_dict):
        return cls(metric_action_dict['action'], metric_action_dict['value'])