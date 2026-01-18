import xcffib
import struct
import io
from . import xproto
from . import render
class randrExtension(xcffib.Extension):

    def QueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', major_version, minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def SetScreenConfig(self, window, timestamp, config_timestamp, sizeID, rotation, rate, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHHH2x', window, timestamp, config_timestamp, sizeID, rotation, rate))
        return self.send_request(2, buf, SetScreenConfigCookie, is_checked=is_checked)

    def SelectInput(self, window, enable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, enable))
        return self.send_request(4, buf, is_checked=is_checked)

    def GetScreenInfo(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(5, buf, GetScreenInfoCookie, is_checked=is_checked)

    def GetScreenSizeRange(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(6, buf, GetScreenSizeRangeCookie, is_checked=is_checked)

    def SetScreenSize(self, window, width, height, mm_width, mm_height, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHII', window, width, height, mm_width, mm_height))
        return self.send_request(7, buf, is_checked=is_checked)

    def GetScreenResources(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(8, buf, GetScreenResourcesCookie, is_checked=is_checked)

    def GetOutputInfo(self, output, config_timestamp, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', output, config_timestamp))
        return self.send_request(9, buf, GetOutputInfoCookie, is_checked=is_checked)

    def ListOutputProperties(self, output, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', output))
        return self.send_request(10, buf, ListOutputPropertiesCookie, is_checked=is_checked)

    def QueryOutputProperty(self, output, property, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', output, property))
        return self.send_request(11, buf, QueryOutputPropertyCookie, is_checked=is_checked)

    def ConfigureOutputProperty(self, output, property, pending, range, values_len, values, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIBB2x', output, property, pending, range))
        buf.write(xcffib.pack_list(values, 'i'))
        return self.send_request(12, buf, is_checked=is_checked)

    def ChangeOutputProperty(self, output, property, type, format, mode, num_units, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIBB2xI', output, property, type, format, mode, num_units))
        buf.write(xcffib.pack_list(data, 'c'))
        return self.send_request(13, buf, is_checked=is_checked)

    def DeleteOutputProperty(self, output, property, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', output, property))
        return self.send_request(14, buf, is_checked=is_checked)

    def GetOutputProperty(self, output, property, type, long_offset, long_length, delete, pending, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIBB2x', output, property, type, long_offset, long_length, delete, pending))
        return self.send_request(15, buf, GetOutputPropertyCookie, is_checked=is_checked)

    def CreateMode(self, window, mode_info, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        buf.write(mode_info.pack() if hasattr(mode_info, 'pack') else ModeInfo.synthetic(*mode_info).pack())
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(16, buf, CreateModeCookie, is_checked=is_checked)

    def DestroyMode(self, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', mode))
        return self.send_request(17, buf, is_checked=is_checked)

    def AddOutputMode(self, output, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', output, mode))
        return self.send_request(18, buf, is_checked=is_checked)

    def DeleteOutputMode(self, output, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', output, mode))
        return self.send_request(19, buf, is_checked=is_checked)

    def GetCrtcInfo(self, crtc, config_timestamp, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', crtc, config_timestamp))
        return self.send_request(20, buf, GetCrtcInfoCookie, is_checked=is_checked)

    def SetCrtcConfig(self, crtc, timestamp, config_timestamp, x, y, mode, rotation, outputs_len, outputs, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhIH2x', crtc, timestamp, config_timestamp, x, y, mode, rotation))
        buf.write(xcffib.pack_list(outputs, 'I'))
        return self.send_request(21, buf, SetCrtcConfigCookie, is_checked=is_checked)

    def GetCrtcGammaSize(self, crtc, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', crtc))
        return self.send_request(22, buf, GetCrtcGammaSizeCookie, is_checked=is_checked)

    def GetCrtcGamma(self, crtc, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', crtc))
        return self.send_request(23, buf, GetCrtcGammaCookie, is_checked=is_checked)

    def SetCrtcGamma(self, crtc, size, red, green, blue, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', crtc, size))
        buf.write(xcffib.pack_list(red, 'H'))
        buf.write(xcffib.pack_list(green, 'H'))
        buf.write(xcffib.pack_list(blue, 'H'))
        return self.send_request(24, buf, is_checked=is_checked)

    def GetScreenResourcesCurrent(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(25, buf, GetScreenResourcesCurrentCookie, is_checked=is_checked)

    def SetCrtcTransform(self, crtc, transform, filter_len, filter_name, filter_params_len, filter_params, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', crtc))
        buf.write(transform.pack() if hasattr(transform, 'pack') else render.TRANSFORM.synthetic(*transform).pack())
        buf.write(struct.pack('=H', filter_len))
        buf.write(struct.pack('=2x'))
        buf.write(xcffib.pack_list(filter_name, 'c'))
        buf.write(struct.pack('=4x'))
        buf.write(xcffib.pack_list(filter_params, 'i'))
        return self.send_request(26, buf, is_checked=is_checked)

    def GetCrtcTransform(self, crtc, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', crtc))
        return self.send_request(27, buf, GetCrtcTransformCookie, is_checked=is_checked)

    def GetPanning(self, crtc, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', crtc))
        return self.send_request(28, buf, GetPanningCookie, is_checked=is_checked)

    def SetPanning(self, crtc, timestamp, left, top, width, height, track_left, track_top, track_width, track_height, border_left, border_top, border_right, border_bottom, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHHHHHhhhh', crtc, timestamp, left, top, width, height, track_left, track_top, track_width, track_height, border_left, border_top, border_right, border_bottom))
        return self.send_request(29, buf, SetPanningCookie, is_checked=is_checked)

    def SetOutputPrimary(self, window, output, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, output))
        return self.send_request(30, buf, is_checked=is_checked)

    def GetOutputPrimary(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(31, buf, GetOutputPrimaryCookie, is_checked=is_checked)

    def GetProviders(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(32, buf, GetProvidersCookie, is_checked=is_checked)

    def GetProviderInfo(self, provider, config_timestamp, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', provider, config_timestamp))
        return self.send_request(33, buf, GetProviderInfoCookie, is_checked=is_checked)

    def SetProviderOffloadSink(self, provider, sink_provider, config_timestamp, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', provider, sink_provider, config_timestamp))
        return self.send_request(34, buf, is_checked=is_checked)

    def SetProviderOutputSource(self, provider, source_provider, config_timestamp, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', provider, source_provider, config_timestamp))
        return self.send_request(35, buf, is_checked=is_checked)

    def ListProviderProperties(self, provider, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', provider))
        return self.send_request(36, buf, ListProviderPropertiesCookie, is_checked=is_checked)

    def QueryProviderProperty(self, provider, property, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', provider, property))
        return self.send_request(37, buf, QueryProviderPropertyCookie, is_checked=is_checked)

    def ConfigureProviderProperty(self, provider, property, pending, range, values_len, values, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIBB2x', provider, property, pending, range))
        buf.write(xcffib.pack_list(values, 'i'))
        return self.send_request(38, buf, is_checked=is_checked)

    def ChangeProviderProperty(self, provider, property, type, format, mode, num_items, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIBB2xI', provider, property, type, format, mode, num_items))
        buf.write(xcffib.pack_list(data, 'c'))
        return self.send_request(39, buf, is_checked=is_checked)

    def DeleteProviderProperty(self, provider, property, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', provider, property))
        return self.send_request(40, buf, is_checked=is_checked)

    def GetProviderProperty(self, provider, property, type, long_offset, long_length, delete, pending, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIBB2x', provider, property, type, long_offset, long_length, delete, pending))
        return self.send_request(41, buf, GetProviderPropertyCookie, is_checked=is_checked)

    def GetMonitors(self, window, get_active, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB', window, get_active))
        return self.send_request(42, buf, GetMonitorsCookie, is_checked=is_checked)

    def SetMonitor(self, window, monitorinfo, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        buf.write(monitorinfo.pack() if hasattr(monitorinfo, 'pack') else MonitorInfo.synthetic(*monitorinfo).pack())
        return self.send_request(43, buf, is_checked=is_checked)

    def DeleteMonitor(self, window, name, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, name))
        return self.send_request(44, buf, is_checked=is_checked)

    def CreateLease(self, window, lid, num_crtcs, num_outputs, crtcs, outputs, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHH', window, lid, num_crtcs, num_outputs))
        buf.write(xcffib.pack_list(crtcs, 'I'))
        buf.write(xcffib.pack_list(outputs, 'I'))
        return self.send_request(45, buf, CreateLeaseCookie, is_checked=is_checked)

    def FreeLease(self, lid, terminate, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB', lid, terminate))
        return self.send_request(46, buf, is_checked=is_checked)