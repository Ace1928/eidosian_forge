from pyparsing import (Word, ZeroOrMore, printables, Suppress, OneOrMore, Group,
def __antlrAlternativeConverter(pyparsingRules, antlrAlternative):
    elementList = []
    for element in antlrAlternative.elements:
        rule = None
        if hasattr(element.atom, 'c1') and element.atom.c1 != '':
            regex = '[' + str(element.atom.c1[0]) + '-' + str(element.atom.c2[0] + ']')
            rule = Regex(regex)('anonymous_regex')
        elif hasattr(element, 'block') and element.block != '':
            rule = __antlrAlternativesConverter(pyparsingRules, element.block)
        else:
            ruleRef = element.atom[0]
            assert ruleRef in pyparsingRules
            rule = pyparsingRules[ruleRef](ruleRef)
        if hasattr(element, 'op') and element.op != '':
            if element.op == '+':
                rule = Group(OneOrMore(rule))('anonymous_one_or_more')
            elif element.op == '*':
                rule = Group(ZeroOrMore(rule))('anonymous_zero_or_more')
            elif element.op == '?':
                rule = Optional(rule)
            else:
                raise Exception('rule operator not yet implemented : ' + element.op)
        rule = rule
        elementList.append(rule)
    if len(elementList) > 1:
        rule = Group(And(elementList))('anonymous_and')
    else:
        rule = elementList[0]
    assert rule is not None
    return rule