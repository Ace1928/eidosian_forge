from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def Equals(self, x):
    if x is self:
        return 1
    if self.has_ts_ != x.has_ts_:
        return 0
    if self.has_ts_ and self.ts_ != x.ts_:
        return 0
    return 1