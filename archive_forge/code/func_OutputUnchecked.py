from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def OutputUnchecked(self, out):
    out.putVarInt32(8)
    out.putVarInt64(self.ts_)