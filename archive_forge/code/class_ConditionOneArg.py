from boto.dynamodb.types import dynamize_value
class ConditionOneArg(Condition):
    """
    Abstract class for Conditions that require a single argument
    such as EQ or NE.
    """

    def __init__(self, v1):
        self.v1 = v1

    def __repr__(self):
        return '%s:%s' % (self.__class__.__name__, self.v1)

    def to_dict(self):
        return {'AttributeValueList': [dynamize_value(self.v1)], 'ComparisonOperator': self.__class__.__name__}