from boto.compat import six
def __first_or_default(self, prop):
    for transition in self:
        return getattr(transition, prop)
    return None