def ZeroOrMore(item, repeat=None):
    result = Optional(OneOrMore(item, repeat))
    return result