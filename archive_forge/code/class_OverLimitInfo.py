from oslo_limit._i18n import _
class OverLimitInfo(object):

    def __init__(self, resource_name, limit, current_usage, delta):
        self.resource_name = resource_name
        self.limit = int(limit)
        self.current_usage = int(current_usage)
        self.delta = int(delta)

    def __str__(self):
        template = 'Resource %s is over limit of %s due to current usage %s and delta %s'
        return template % (self.resource_name, self.limit, self.current_usage, self.delta)

    def __repr__(self):
        return self.__str__()