from pyparsing import (Word, ZeroOrMore, printables, Suppress, OneOrMore, Group,
def __antlrRuleConverter(pyparsingRules, antlrRule):
    rule = None
    rule = __antlrAlternativesConverter(pyparsingRules, antlrRule)
    assert rule != None
    rule(antlrRule.ruleName)
    return rule