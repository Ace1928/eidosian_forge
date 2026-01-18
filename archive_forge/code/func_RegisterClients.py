import xcffib
import struct
import io
def RegisterClients(self, context, element_header, num_client_specs, num_ranges, client_specs, ranges, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3xII', context, element_header, num_client_specs, num_ranges))
    buf.write(xcffib.pack_list(client_specs, 'I'))
    buf.write(xcffib.pack_list(ranges, Range))
    return self.send_request(2, buf, is_checked=is_checked)