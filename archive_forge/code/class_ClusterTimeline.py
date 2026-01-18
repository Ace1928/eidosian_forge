from boto.resultset import ResultSet
class ClusterTimeline(EmrObject):
    Fields = set(['CreationDateTime', 'ReadyDateTime', 'EndDateTime'])