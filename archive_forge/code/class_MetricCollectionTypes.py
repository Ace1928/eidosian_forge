from boto.resultset import ResultSet
from boto.ec2.elb.listelement import ListElement
class MetricCollectionTypes(object):

    class BaseType(object):
        arg = ''

        def __init__(self, connection):
            self.connection = connection
            self.val = None

        def __repr__(self):
            return '%s:%s' % (self.arg, self.val)

        def startElement(self, name, attrs, connection):
            return

        def endElement(self, name, value, connection):
            if name == self.arg:
                self.val = value

    class Metric(BaseType):
        arg = 'Metric'

    class Granularity(BaseType):
        arg = 'Granularity'

    def __init__(self, connection=None):
        self.connection = connection
        self.metrics = []
        self.granularities = []

    def __repr__(self):
        return 'MetricCollectionTypes:<%s, %s>' % (self.metrics, self.granularities)

    def startElement(self, name, attrs, connection):
        if name == 'Granularities':
            self.granularities = ResultSet([('member', self.Granularity)])
            return self.granularities
        elif name == 'Metrics':
            self.metrics = ResultSet([('member', self.Metric)])
            return self.metrics

    def endElement(self, name, value, connection):
        return