import xcffib
import struct
import io
class xevieExtension(xcffib.Extension):

    def QueryVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', client_major_version, client_minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def Start(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(1, buf, StartCookie, is_checked=is_checked)

    def End(self, cmap, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        return self.send_request(2, buf, EndCookie, is_checked=is_checked)

    def Send(self, event, data_type, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        buf.write(event.pack() if hasattr(event, 'pack') else Event.synthetic(*event).pack())
        buf.write(struct.pack('=I', data_type))
        buf.write(struct.pack('=64x'))
        return self.send_request(3, buf, SendCookie, is_checked=is_checked)

    def SelectInput(self, event_mask, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', event_mask))
        return self.send_request(4, buf, SelectInputCookie, is_checked=is_checked)