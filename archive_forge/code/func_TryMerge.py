from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def TryMerge(self, d):
    while d.avail() > 0:
        tt = d.getVarInt32()
        if tt == 8:
            self.set_ts(d.getVarInt64())
            continue
        if tt == 0:
            raise ProtocolBuffer.ProtocolBufferDecodeError()
        d.skipData(tt)