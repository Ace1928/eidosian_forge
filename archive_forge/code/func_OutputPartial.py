from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def OutputPartial(self, out):
    if self.has_ts_:
        out.putVarInt32(8)
        out.putVarInt64(self.ts_)