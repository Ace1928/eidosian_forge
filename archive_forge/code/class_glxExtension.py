import xcffib
import struct
import io
from . import xproto
class glxExtension(xcffib.Extension):

    def Render(self, context_tag, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(1, buf, is_checked=is_checked)

    def RenderLarge(self, context_tag, request_num, request_total, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHI', context_tag, request_num, request_total, data_len))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(2, buf, is_checked=is_checked)

    def CreateContext(self, context, visual, screen, share_list, is_direct, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIB3x', context, visual, screen, share_list, is_direct))
        return self.send_request(3, buf, is_checked=is_checked)

    def DestroyContext(self, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context))
        return self.send_request(4, buf, is_checked=is_checked)

    def MakeCurrent(self, drawable, context, old_context_tag, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', drawable, context, old_context_tag))
        return self.send_request(5, buf, MakeCurrentCookie, is_checked=is_checked)

    def IsDirect(self, context, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context))
        return self.send_request(6, buf, IsDirectCookie, is_checked=is_checked)

    def QueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', major_version, minor_version))
        return self.send_request(7, buf, QueryVersionCookie, is_checked=is_checked)

    def WaitGL(self, context_tag, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(8, buf, is_checked=is_checked)

    def WaitX(self, context_tag, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(9, buf, is_checked=is_checked)

    def CopyContext(self, src, dest, mask, src_context_tag, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIII', src, dest, mask, src_context_tag))
        return self.send_request(10, buf, is_checked=is_checked)

    def SwapBuffers(self, context_tag, drawable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, drawable))
        return self.send_request(11, buf, is_checked=is_checked)

    def UseXFont(self, context_tag, font, first, count, list_base, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', context_tag, font, first, count, list_base))
        return self.send_request(12, buf, is_checked=is_checked)

    def CreateGLXPixmap(self, screen, visual, pixmap, glx_pixmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIII', screen, visual, pixmap, glx_pixmap))
        return self.send_request(13, buf, is_checked=is_checked)

    def GetVisualConfigs(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(14, buf, GetVisualConfigsCookie, is_checked=is_checked)

    def DestroyGLXPixmap(self, glx_pixmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', glx_pixmap))
        return self.send_request(15, buf, is_checked=is_checked)

    def VendorPrivate(self, vendor_code, context_tag, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', vendor_code, context_tag))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(16, buf, is_checked=is_checked)

    def VendorPrivateWithReply(self, vendor_code, context_tag, data_len, data, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', vendor_code, context_tag))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(17, buf, VendorPrivateWithReplyCookie, is_checked=is_checked)

    def QueryExtensionsString(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(18, buf, QueryExtensionsStringCookie, is_checked=is_checked)

    def QueryServerString(self, screen, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, name))
        return self.send_request(19, buf, QueryServerStringCookie, is_checked=is_checked)

    def ClientInfo(self, major_version, minor_version, str_len, string, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', major_version, minor_version, str_len))
        buf.write(xcffib.pack_list(string, 'c'))
        return self.send_request(20, buf, is_checked=is_checked)

    def GetFBConfigs(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(21, buf, GetFBConfigsCookie, is_checked=is_checked)

    def CreatePixmap(self, screen, fbconfig, pixmap, glx_pixmap, num_attribs, attribs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', screen, fbconfig, pixmap, glx_pixmap, num_attribs))
        buf.write(xcffib.pack_list(attribs, 'I'))
        return self.send_request(22, buf, is_checked=is_checked)

    def DestroyPixmap(self, glx_pixmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', glx_pixmap))
        return self.send_request(23, buf, is_checked=is_checked)

    def CreateNewContext(self, context, fbconfig, screen, render_type, share_list, is_direct, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIB3x', context, fbconfig, screen, render_type, share_list, is_direct))
        return self.send_request(24, buf, is_checked=is_checked)

    def QueryContext(self, context, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context))
        return self.send_request(25, buf, QueryContextCookie, is_checked=is_checked)

    def MakeContextCurrent(self, old_context_tag, drawable, read_drawable, context, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIII', old_context_tag, drawable, read_drawable, context))
        return self.send_request(26, buf, MakeContextCurrentCookie, is_checked=is_checked)

    def CreatePbuffer(self, screen, fbconfig, pbuffer, num_attribs, attribs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIII', screen, fbconfig, pbuffer, num_attribs))
        buf.write(xcffib.pack_list(attribs, 'I'))
        return self.send_request(27, buf, is_checked=is_checked)

    def DestroyPbuffer(self, pbuffer, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', pbuffer))
        return self.send_request(28, buf, is_checked=is_checked)

    def GetDrawableAttributes(self, drawable, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', drawable))
        return self.send_request(29, buf, GetDrawableAttributesCookie, is_checked=is_checked)

    def ChangeDrawableAttributes(self, drawable, num_attribs, attribs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, num_attribs))
        buf.write(xcffib.pack_list(attribs, 'I'))
        return self.send_request(30, buf, is_checked=is_checked)

    def CreateWindow(self, screen, fbconfig, window, glx_window, num_attribs, attribs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', screen, fbconfig, window, glx_window, num_attribs))
        buf.write(xcffib.pack_list(attribs, 'I'))
        return self.send_request(31, buf, is_checked=is_checked)

    def DeleteWindow(self, glxwindow, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', glxwindow))
        return self.send_request(32, buf, is_checked=is_checked)

    def SetClientInfoARB(self, major_version, minor_version, num_versions, gl_str_len, glx_str_len, gl_versions, gl_extension_string, glx_extension_string, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', major_version, minor_version, num_versions, gl_str_len, glx_str_len))
        buf.write(xcffib.pack_list(gl_versions, 'I'))
        buf.write(xcffib.pack_list(gl_extension_string, 'c'))
        buf.write(struct.pack('=4x'))
        buf.write(xcffib.pack_list(glx_extension_string, 'c'))
        return self.send_request(33, buf, is_checked=is_checked)

    def CreateContextAttribsARB(self, context, fbconfig, screen, share_list, is_direct, num_attribs, attribs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIB3xI', context, fbconfig, screen, share_list, is_direct, num_attribs))
        buf.write(xcffib.pack_list(attribs, 'I'))
        return self.send_request(34, buf, is_checked=is_checked)

    def SetClientInfo2ARB(self, major_version, minor_version, num_versions, gl_str_len, glx_str_len, gl_versions, gl_extension_string, glx_extension_string, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', major_version, minor_version, num_versions, gl_str_len, glx_str_len))
        buf.write(xcffib.pack_list(gl_versions, 'I'))
        buf.write(xcffib.pack_list(gl_extension_string, 'c'))
        buf.write(struct.pack('=4x'))
        buf.write(xcffib.pack_list(glx_extension_string, 'c'))
        return self.send_request(35, buf, is_checked=is_checked)

    def NewList(self, context_tag, list, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, list, mode))
        return self.send_request(101, buf, is_checked=is_checked)

    def EndList(self, context_tag, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(102, buf, is_checked=is_checked)

    def DeleteLists(self, context_tag, list, range, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIi', context_tag, list, range))
        return self.send_request(103, buf, is_checked=is_checked)

    def GenLists(self, context_tag, range, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, range))
        return self.send_request(104, buf, GenListsCookie, is_checked=is_checked)

    def FeedbackBuffer(self, context_tag, size, type, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIii', context_tag, size, type))
        return self.send_request(105, buf, is_checked=is_checked)

    def SelectBuffer(self, context_tag, size, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, size))
        return self.send_request(106, buf, is_checked=is_checked)

    def RenderMode(self, context_tag, mode, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, mode))
        return self.send_request(107, buf, RenderModeCookie, is_checked=is_checked)

    def Finish(self, context_tag, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(108, buf, FinishCookie, is_checked=is_checked)

    def PixelStoref(self, context_tag, pname, datum, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIf', context_tag, pname, datum))
        return self.send_request(109, buf, is_checked=is_checked)

    def PixelStorei(self, context_tag, pname, datum, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIi', context_tag, pname, datum))
        return self.send_request(110, buf, is_checked=is_checked)

    def ReadPixels(self, context_tag, x, y, width, height, format, type, swap_bytes, lsb_first, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIiiiiIIBB', context_tag, x, y, width, height, format, type, swap_bytes, lsb_first))
        return self.send_request(111, buf, ReadPixelsCookie, is_checked=is_checked)

    def GetBooleanv(self, context_tag, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, pname))
        return self.send_request(112, buf, GetBooleanvCookie, is_checked=is_checked)

    def GetClipPlane(self, context_tag, plane, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, plane))
        return self.send_request(113, buf, is_checked=is_checked)

    def GetDoublev(self, context_tag, pname, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, pname))
        return self.send_request(114, buf, is_checked=is_checked)

    def GetError(self, context_tag, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(115, buf, GetErrorCookie, is_checked=is_checked)

    def GetFloatv(self, context_tag, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, pname))
        return self.send_request(116, buf, GetFloatvCookie, is_checked=is_checked)

    def GetIntegerv(self, context_tag, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, pname))
        return self.send_request(117, buf, GetIntegervCookie, is_checked=is_checked)

    def GetLightfv(self, context_tag, light, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, light, pname))
        return self.send_request(118, buf, GetLightfvCookie, is_checked=is_checked)

    def GetLightiv(self, context_tag, light, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, light, pname))
        return self.send_request(119, buf, GetLightivCookie, is_checked=is_checked)

    def GetMapdv(self, context_tag, target, query, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, query))
        return self.send_request(120, buf, is_checked=is_checked)

    def GetMapfv(self, context_tag, target, query, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, query))
        return self.send_request(121, buf, GetMapfvCookie, is_checked=is_checked)

    def GetMapiv(self, context_tag, target, query, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, query))
        return self.send_request(122, buf, GetMapivCookie, is_checked=is_checked)

    def GetMaterialfv(self, context_tag, face, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, face, pname))
        return self.send_request(123, buf, GetMaterialfvCookie, is_checked=is_checked)

    def GetMaterialiv(self, context_tag, face, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, face, pname))
        return self.send_request(124, buf, GetMaterialivCookie, is_checked=is_checked)

    def GetPixelMapfv(self, context_tag, map, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, map))
        return self.send_request(125, buf, GetPixelMapfvCookie, is_checked=is_checked)

    def GetPixelMapuiv(self, context_tag, map, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, map))
        return self.send_request(126, buf, GetPixelMapuivCookie, is_checked=is_checked)

    def GetPixelMapusv(self, context_tag, map, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, map))
        return self.send_request(127, buf, GetPixelMapusvCookie, is_checked=is_checked)

    def GetPolygonStipple(self, context_tag, lsb_first, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB', context_tag, lsb_first))
        return self.send_request(128, buf, GetPolygonStippleCookie, is_checked=is_checked)

    def GetString(self, context_tag, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, name))
        return self.send_request(129, buf, GetStringCookie, is_checked=is_checked)

    def GetTexEnvfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(130, buf, GetTexEnvfvCookie, is_checked=is_checked)

    def GetTexEnviv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(131, buf, GetTexEnvivCookie, is_checked=is_checked)

    def GetTexGendv(self, context_tag, coord, pname, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, coord, pname))
        return self.send_request(132, buf, is_checked=is_checked)

    def GetTexGenfv(self, context_tag, coord, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, coord, pname))
        return self.send_request(133, buf, GetTexGenfvCookie, is_checked=is_checked)

    def GetTexGeniv(self, context_tag, coord, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, coord, pname))
        return self.send_request(134, buf, GetTexGenivCookie, is_checked=is_checked)

    def GetTexImage(self, context_tag, target, level, format, type, swap_bytes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIiIIB', context_tag, target, level, format, type, swap_bytes))
        return self.send_request(135, buf, GetTexImageCookie, is_checked=is_checked)

    def GetTexParameterfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(136, buf, GetTexParameterfvCookie, is_checked=is_checked)

    def GetTexParameteriv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(137, buf, GetTexParameterivCookie, is_checked=is_checked)

    def GetTexLevelParameterfv(self, context_tag, target, level, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIiI', context_tag, target, level, pname))
        return self.send_request(138, buf, GetTexLevelParameterfvCookie, is_checked=is_checked)

    def GetTexLevelParameteriv(self, context_tag, target, level, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIiI', context_tag, target, level, pname))
        return self.send_request(139, buf, GetTexLevelParameterivCookie, is_checked=is_checked)

    def IsEnabled(self, context_tag, capability, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, capability))
        return self.send_request(140, buf, IsEnabledCookie, is_checked=is_checked)

    def IsList(self, context_tag, list, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, list))
        return self.send_request(141, buf, IsListCookie, is_checked=is_checked)

    def Flush(self, context_tag, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_tag))
        return self.send_request(142, buf, is_checked=is_checked)

    def AreTexturesResident(self, context_tag, n, textures, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, n))
        buf.write(xcffib.pack_list(textures, 'I'))
        return self.send_request(143, buf, AreTexturesResidentCookie, is_checked=is_checked)

    def DeleteTextures(self, context_tag, n, textures, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, n))
        buf.write(xcffib.pack_list(textures, 'I'))
        return self.send_request(144, buf, is_checked=is_checked)

    def GenTextures(self, context_tag, n, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, n))
        return self.send_request(145, buf, GenTexturesCookie, is_checked=is_checked)

    def IsTexture(self, context_tag, texture, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, texture))
        return self.send_request(146, buf, IsTextureCookie, is_checked=is_checked)

    def GetColorTable(self, context_tag, target, format, type, swap_bytes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIB', context_tag, target, format, type, swap_bytes))
        return self.send_request(147, buf, GetColorTableCookie, is_checked=is_checked)

    def GetColorTableParameterfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(148, buf, GetColorTableParameterfvCookie, is_checked=is_checked)

    def GetColorTableParameteriv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(149, buf, GetColorTableParameterivCookie, is_checked=is_checked)

    def GetConvolutionFilter(self, context_tag, target, format, type, swap_bytes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIB', context_tag, target, format, type, swap_bytes))
        return self.send_request(150, buf, GetConvolutionFilterCookie, is_checked=is_checked)

    def GetConvolutionParameterfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(151, buf, GetConvolutionParameterfvCookie, is_checked=is_checked)

    def GetConvolutionParameteriv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(152, buf, GetConvolutionParameterivCookie, is_checked=is_checked)

    def GetSeparableFilter(self, context_tag, target, format, type, swap_bytes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIB', context_tag, target, format, type, swap_bytes))
        return self.send_request(153, buf, GetSeparableFilterCookie, is_checked=is_checked)

    def GetHistogram(self, context_tag, target, format, type, swap_bytes, reset, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIBB', context_tag, target, format, type, swap_bytes, reset))
        return self.send_request(154, buf, GetHistogramCookie, is_checked=is_checked)

    def GetHistogramParameterfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(155, buf, GetHistogramParameterfvCookie, is_checked=is_checked)

    def GetHistogramParameteriv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(156, buf, GetHistogramParameterivCookie, is_checked=is_checked)

    def GetMinmax(self, context_tag, target, format, type, swap_bytes, reset, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIBB', context_tag, target, format, type, swap_bytes, reset))
        return self.send_request(157, buf, GetMinmaxCookie, is_checked=is_checked)

    def GetMinmaxParameterfv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(158, buf, GetMinmaxParameterfvCookie, is_checked=is_checked)

    def GetMinmaxParameteriv(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(159, buf, GetMinmaxParameterivCookie, is_checked=is_checked)

    def GetCompressedTexImageARB(self, context_tag, target, level, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIi', context_tag, target, level))
        return self.send_request(160, buf, GetCompressedTexImageARBCookie, is_checked=is_checked)

    def DeleteQueriesARB(self, context_tag, n, ids, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, n))
        buf.write(xcffib.pack_list(ids, 'I'))
        return self.send_request(161, buf, is_checked=is_checked)

    def GenQueriesARB(self, context_tag, n, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', context_tag, n))
        return self.send_request(162, buf, GenQueriesARBCookie, is_checked=is_checked)

    def IsQueryARB(self, context_tag, id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', context_tag, id))
        return self.send_request(163, buf, IsQueryARBCookie, is_checked=is_checked)

    def GetQueryivARB(self, context_tag, target, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, target, pname))
        return self.send_request(164, buf, GetQueryivARBCookie, is_checked=is_checked)

    def GetQueryObjectivARB(self, context_tag, id, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, id, pname))
        return self.send_request(165, buf, GetQueryObjectivARBCookie, is_checked=is_checked)

    def GetQueryObjectuivARB(self, context_tag, id, pname, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', context_tag, id, pname))
        return self.send_request(166, buf, GetQueryObjectuivARBCookie, is_checked=is_checked)