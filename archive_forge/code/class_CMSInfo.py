from ..helpers import nativestr
class CMSInfo(object):
    width = None
    depth = None
    count = None

    def __init__(self, args):
        response = dict(zip(map(nativestr, args[::2]), args[1::2]))
        self.width = response['width']
        self.depth = response['depth']
        self.count = response['count']

    def __getitem__(self, item):
        return getattr(self, item)