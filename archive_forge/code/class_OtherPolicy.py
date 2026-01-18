from boto.resultset import ResultSet
class OtherPolicy(object):

    def __init__(self, connection=None):
        self.policy_name = None

    def __repr__(self):
        return 'OtherPolicy(%s)' % self.policy_name

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        self.policy_name = value