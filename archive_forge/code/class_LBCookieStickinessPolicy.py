from boto.resultset import ResultSet
class LBCookieStickinessPolicy(object):

    def __init__(self, connection=None):
        self.policy_name = None
        self.cookie_expiration_period = None

    def __repr__(self):
        return 'LBCookieStickiness(%s, %s)' % (self.policy_name, self.cookie_expiration_period)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'CookieExpirationPeriod':
            self.cookie_expiration_period = value
        elif name == 'PolicyName':
            self.policy_name = value