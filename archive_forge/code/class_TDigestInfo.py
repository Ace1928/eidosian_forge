from ..helpers import nativestr
class TDigestInfo(object):
    compression = None
    capacity = None
    merged_nodes = None
    unmerged_nodes = None
    merged_weight = None
    unmerged_weight = None
    total_compressions = None
    memory_usage = None

    def __init__(self, args):
        response = dict(zip(map(nativestr, args[::2]), args[1::2]))
        self.compression = response['Compression']
        self.capacity = response['Capacity']
        self.merged_nodes = response['Merged nodes']
        self.unmerged_nodes = response['Unmerged nodes']
        self.merged_weight = response['Merged weight']
        self.unmerged_weight = response['Unmerged weight']
        self.total_compressions = response['Total compressions']
        self.memory_usage = response['Memory usage']

    def get(self, item):
        try:
            return self.__getitem__(item)
        except AttributeError:
            return None

    def __getitem__(self, item):
        return getattr(self, item)