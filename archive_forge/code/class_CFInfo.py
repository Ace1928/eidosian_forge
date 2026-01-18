from ..helpers import nativestr
class CFInfo(object):
    size = None
    bucketNum = None
    filterNum = None
    insertedNum = None
    deletedNum = None
    bucketSize = None
    expansionRate = None
    maxIteration = None

    def __init__(self, args):
        response = dict(zip(map(nativestr, args[::2]), args[1::2]))
        self.size = response['Size']
        self.bucketNum = response['Number of buckets']
        self.filterNum = response['Number of filters']
        self.insertedNum = response['Number of items inserted']
        self.deletedNum = response['Number of items deleted']
        self.bucketSize = response['Bucket size']
        self.expansionRate = response['Expansion rate']
        self.maxIteration = response['Max iterations']

    def get(self, item):
        try:
            return self.__getitem__(item)
        except AttributeError:
            return None

    def __getitem__(self, item):
        return getattr(self, item)