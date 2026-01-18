from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
def AddMatch(self, rule):
    """*rule* can be a str or a :class:`MatchRule` instance"""
    if isinstance(rule, MatchRule):
        rule = rule.serialise()
    return new_method_call(self, 'AddMatch', 's', (rule,))