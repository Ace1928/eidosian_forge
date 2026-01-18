from boto.resultset import ResultSet
class ClusterSummaryList(EmrObject):
    Fields = set(['Marker'])

    def __init__(self, connection):
        self.connection = connection
        self.clusters = None

    def startElement(self, name, attrs, connection):
        if name == 'Clusters':
            self.clusters = ResultSet([('member', ClusterSummary)])
            return self.clusters
        else:
            return None