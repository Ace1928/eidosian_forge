from boto.dynamodb.types import dynamize_value
class ConditionTwoArgs(Condition):
    """
    Abstract class for Conditions that require two arguments.
    The only example of this currently is BETWEEN.
    """

    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.v1, self.v2)

    def to_dict(self):
        values = (self.v1, self.v2)
        return {'AttributeValueList': [dynamize_value(v) for v in values], 'ComparisonOperator': self.__class__.__name__}