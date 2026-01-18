import xcffib
import struct
import io
class Setup(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.protocol_major_version, self.protocol_minor_version, self.length, self.release_number, self.resource_id_base, self.resource_id_mask, self.motion_buffer_size, self.vendor_len, self.maximum_request_length, self.roots_len, self.pixmap_formats_len, self.image_byte_order, self.bitmap_format_bit_order, self.bitmap_format_scanline_unit, self.bitmap_format_scanline_pad, self.min_keycode, self.max_keycode = unpacker.unpack('BxHHHIIIIHHBBBBBBBB4x')
        self.vendor = xcffib.List(unpacker, 'c', self.vendor_len)
        unpacker.pad(FORMAT)
        self.pixmap_formats = xcffib.List(unpacker, FORMAT, self.pixmap_formats_len)
        unpacker.pad(SCREEN)
        self.roots = xcffib.List(unpacker, SCREEN, self.roots_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BxHHHIIIIHHBBBBBBBB4x', self.status, self.protocol_major_version, self.protocol_minor_version, self.length, self.release_number, self.resource_id_base, self.resource_id_mask, self.motion_buffer_size, self.vendor_len, self.maximum_request_length, self.roots_len, self.pixmap_formats_len, self.image_byte_order, self.bitmap_format_bit_order, self.bitmap_format_scanline_unit, self.bitmap_format_scanline_pad, self.min_keycode, self.max_keycode))
        buf.write(xcffib.pack_list(self.vendor, 'c'))
        buf.write(struct.pack('=4x'))
        buf.write(xcffib.pack_list(self.pixmap_formats, FORMAT))
        buf.write(xcffib.pack_list(self.roots, SCREEN))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, status, protocol_major_version, protocol_minor_version, length, release_number, resource_id_base, resource_id_mask, motion_buffer_size, vendor_len, maximum_request_length, roots_len, pixmap_formats_len, image_byte_order, bitmap_format_bit_order, bitmap_format_scanline_unit, bitmap_format_scanline_pad, min_keycode, max_keycode, vendor, pixmap_formats, roots):
        self = cls.__new__(cls)
        self.status = status
        self.protocol_major_version = protocol_major_version
        self.protocol_minor_version = protocol_minor_version
        self.length = length
        self.release_number = release_number
        self.resource_id_base = resource_id_base
        self.resource_id_mask = resource_id_mask
        self.motion_buffer_size = motion_buffer_size
        self.vendor_len = vendor_len
        self.maximum_request_length = maximum_request_length
        self.roots_len = roots_len
        self.pixmap_formats_len = pixmap_formats_len
        self.image_byte_order = image_byte_order
        self.bitmap_format_bit_order = bitmap_format_bit_order
        self.bitmap_format_scanline_unit = bitmap_format_scanline_unit
        self.bitmap_format_scanline_pad = bitmap_format_scanline_pad
        self.min_keycode = min_keycode
        self.max_keycode = max_keycode
        self.vendor = vendor
        self.pixmap_formats = pixmap_formats
        self.roots = roots
        return self