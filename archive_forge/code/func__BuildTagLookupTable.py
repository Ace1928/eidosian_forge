from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def _BuildTagLookupTable(sparse, maxtag, default=None):
    return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])