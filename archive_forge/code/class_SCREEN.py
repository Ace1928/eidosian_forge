import xcffib
import struct
import io
class SCREEN(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.root, self.default_colormap, self.white_pixel, self.black_pixel, self.current_input_masks, self.width_in_pixels, self.height_in_pixels, self.width_in_millimeters, self.height_in_millimeters, self.min_installed_maps, self.max_installed_maps, self.root_visual, self.backing_stores, self.save_unders, self.root_depth, self.allowed_depths_len = unpacker.unpack('IIIIIHHHHHHIBBBB')
        self.allowed_depths = xcffib.List(unpacker, DEPTH, self.allowed_depths_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IIIIIHHHHHHIBBBB', self.root, self.default_colormap, self.white_pixel, self.black_pixel, self.current_input_masks, self.width_in_pixels, self.height_in_pixels, self.width_in_millimeters, self.height_in_millimeters, self.min_installed_maps, self.max_installed_maps, self.root_visual, self.backing_stores, self.save_unders, self.root_depth, self.allowed_depths_len))
        buf.write(xcffib.pack_list(self.allowed_depths, DEPTH))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, root, default_colormap, white_pixel, black_pixel, current_input_masks, width_in_pixels, height_in_pixels, width_in_millimeters, height_in_millimeters, min_installed_maps, max_installed_maps, root_visual, backing_stores, save_unders, root_depth, allowed_depths_len, allowed_depths):
        self = cls.__new__(cls)
        self.root = root
        self.default_colormap = default_colormap
        self.white_pixel = white_pixel
        self.black_pixel = black_pixel
        self.current_input_masks = current_input_masks
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_millimeters = width_in_millimeters
        self.height_in_millimeters = height_in_millimeters
        self.min_installed_maps = min_installed_maps
        self.max_installed_maps = max_installed_maps
        self.root_visual = root_visual
        self.backing_stores = backing_stores
        self.save_unders = save_unders
        self.root_depth = root_depth
        self.allowed_depths_len = allowed_depths_len
        self.allowed_depths = allowed_depths
        return self