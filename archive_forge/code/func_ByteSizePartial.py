from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def ByteSizePartial(self):
    n = 0
    if self.has_ts_:
        n += 1
        n += self.lengthVarInt64(self.ts_)
    return n