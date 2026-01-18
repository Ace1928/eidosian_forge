import xcffib
import struct
import io
from . import xfixes
from . import xproto
class xinputExtension(xcffib.Extension):

    def GetExtensionVersion(self, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(1, buf, GetExtensionVersionCookie, is_checked=is_checked)

    def ListInputDevices(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(2, buf, ListInputDevicesCookie, is_checked=is_checked)

    def OpenDevice(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(3, buf, OpenDeviceCookie, is_checked=is_checked)

    def CloseDevice(self, device_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(4, buf, is_checked=is_checked)

    def SetDeviceMode(self, device_id, mode, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2x', device_id, mode))
        return self.send_request(5, buf, SetDeviceModeCookie, is_checked=is_checked)

    def SelectExtensionEvent(self, window, num_classes, classes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, num_classes))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(6, buf, is_checked=is_checked)

    def GetSelectedExtensionEvents(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(7, buf, GetSelectedExtensionEventsCookie, is_checked=is_checked)

    def ChangeDeviceDontPropagateList(self, window, num_classes, mode, classes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHBx', window, num_classes, mode))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(8, buf, is_checked=is_checked)

    def GetDeviceDontPropagateList(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(9, buf, GetDeviceDontPropagateListCookie, is_checked=is_checked)

    def GetDeviceMotionEvents(self, start, stop, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', start, stop, device_id))
        return self.send_request(10, buf, GetDeviceMotionEventsCookie, is_checked=is_checked)

    def ChangeKeyboardDevice(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(11, buf, ChangeKeyboardDeviceCookie, is_checked=is_checked)

    def ChangePointerDevice(self, x_axis, y_axis, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBx', x_axis, y_axis, device_id))
        return self.send_request(12, buf, ChangePointerDeviceCookie, is_checked=is_checked)

    def GrabDevice(self, grab_window, time, num_classes, this_device_mode, other_device_mode, owner_events, device_id, classes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHBBBB2x', grab_window, time, num_classes, this_device_mode, other_device_mode, owner_events, device_id))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(13, buf, GrabDeviceCookie, is_checked=is_checked)

    def UngrabDevice(self, time, device_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', time, device_id))
        return self.send_request(14, buf, is_checked=is_checked)

    def GrabDeviceKey(self, grab_window, num_classes, modifiers, modifier_device, grabbed_device, key, this_device_mode, other_device_mode, owner_events, classes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHBBBBBB2x', grab_window, num_classes, modifiers, modifier_device, grabbed_device, key, this_device_mode, other_device_mode, owner_events))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(15, buf, is_checked=is_checked)

    def UngrabDeviceKey(self, grabWindow, modifiers, modifier_device, key, grabbed_device, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHBBB', grabWindow, modifiers, modifier_device, key, grabbed_device))
        return self.send_request(16, buf, is_checked=is_checked)

    def GrabDeviceButton(self, grab_window, grabbed_device, modifier_device, num_classes, modifiers, this_device_mode, other_device_mode, button, owner_events, classes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIBBHHBBBB2x', grab_window, grabbed_device, modifier_device, num_classes, modifiers, this_device_mode, other_device_mode, button, owner_events))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(17, buf, is_checked=is_checked)

    def UngrabDeviceButton(self, grab_window, modifiers, modifier_device, button, grabbed_device, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHBBB3x', grab_window, modifiers, modifier_device, button, grabbed_device))
        return self.send_request(18, buf, is_checked=is_checked)

    def AllowDeviceEvents(self, time, mode, device_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIBB2x', time, mode, device_id))
        return self.send_request(19, buf, is_checked=is_checked)

    def GetDeviceFocus(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(20, buf, GetDeviceFocusCookie, is_checked=is_checked)

    def SetDeviceFocus(self, focus, time, revert_to, device_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIBB2x', focus, time, revert_to, device_id))
        return self.send_request(21, buf, is_checked=is_checked)

    def GetFeedbackControl(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(22, buf, GetFeedbackControlCookie, is_checked=is_checked)

    def ChangeFeedbackControl(self, mask, device_id, feedback_id, feedback, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIBB2x', mask, device_id, feedback_id))
        buf.write(feedback.pack() if hasattr(feedback, 'pack') else FeedbackCtl.synthetic(*feedback).pack())
        return self.send_request(23, buf, is_checked=is_checked)

    def GetDeviceKeyMapping(self, device_id, first_keycode, count, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBx', device_id, first_keycode, count))
        return self.send_request(24, buf, GetDeviceKeyMappingCookie, is_checked=is_checked)

    def ChangeDeviceKeyMapping(self, device_id, first_keycode, keysyms_per_keycode, keycode_count, keysyms, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBB', device_id, first_keycode, keysyms_per_keycode, keycode_count))
        buf.write(xcffib.pack_list(keysyms, 'I'))
        return self.send_request(25, buf, is_checked=is_checked)

    def GetDeviceModifierMapping(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(26, buf, GetDeviceModifierMappingCookie, is_checked=is_checked)

    def SetDeviceModifierMapping(self, device_id, keycodes_per_modifier, keymaps, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2x', device_id, keycodes_per_modifier))
        buf.write(xcffib.pack_list(keymaps, 'B'))
        return self.send_request(27, buf, SetDeviceModifierMappingCookie, is_checked=is_checked)

    def GetDeviceButtonMapping(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(28, buf, GetDeviceButtonMappingCookie, is_checked=is_checked)

    def SetDeviceButtonMapping(self, device_id, map_size, map, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2x', device_id, map_size))
        buf.write(xcffib.pack_list(map, 'B'))
        return self.send_request(29, buf, SetDeviceButtonMappingCookie, is_checked=is_checked)

    def QueryDeviceState(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(30, buf, QueryDeviceStateCookie, is_checked=is_checked)

    def DeviceBell(self, device_id, feedback_id, feedback_class, percent, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBb', device_id, feedback_id, feedback_class, percent))
        return self.send_request(32, buf, is_checked=is_checked)

    def SetDeviceValuators(self, device_id, first_valuator, num_valuators, valuators, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBx', device_id, first_valuator, num_valuators))
        buf.write(xcffib.pack_list(valuators, 'i'))
        return self.send_request(33, buf, SetDeviceValuatorsCookie, is_checked=is_checked)

    def GetDeviceControl(self, control_id, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHBx', control_id, device_id))
        return self.send_request(34, buf, GetDeviceControlCookie, is_checked=is_checked)

    def ChangeDeviceControl(self, control_id, device_id, control, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHBx', control_id, device_id))
        buf.write(control.pack() if hasattr(control, 'pack') else DeviceCtl.synthetic(*control).pack())
        return self.send_request(35, buf, ChangeDeviceControlCookie, is_checked=is_checked)

    def ListDeviceProperties(self, device_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', device_id))
        return self.send_request(36, buf, ListDevicePropertiesCookie, is_checked=is_checked)

    def ChangeDeviceProperty(self, property, type, device_id, format, mode, num_items, items, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIBBBxI', property, type, device_id, format, mode, num_items))
        if format & PropertyFormat._8Bits:
            data8 = items.pop(0)
            items.pop(0)
            buf.write(xcffib.pack_list(data8, 'B'))
            buf.write(struct.pack('=4x'))
        if format & PropertyFormat._16Bits:
            data16 = items.pop(0)
            items.pop(0)
            buf.write(xcffib.pack_list(data16, 'H'))
            buf.write(struct.pack('=4x'))
        if format & PropertyFormat._32Bits:
            data32 = items.pop(0)
            buf.write(xcffib.pack_list(data32, 'I'))
        return self.send_request(37, buf, is_checked=is_checked)

    def DeleteDeviceProperty(self, property, device_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', property, device_id))
        return self.send_request(38, buf, is_checked=is_checked)

    def GetDeviceProperty(self, property, type, offset, len, device_id, delete, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIBB2x', property, type, offset, len, device_id, delete))
        return self.send_request(39, buf, GetDevicePropertyCookie, is_checked=is_checked)

    def XIQueryPointer(self, window, deviceid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, deviceid))
        return self.send_request(40, buf, XIQueryPointerCookie, is_checked=is_checked)

    def XIWarpPointer(self, src_win, dst_win, src_x, src_y, src_width, src_height, dst_x, dst_y, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIiiHHiiH2x', src_win, dst_win, src_x, src_y, src_width, src_height, dst_x, dst_y, deviceid))
        return self.send_request(41, buf, is_checked=is_checked)

    def XIChangeCursor(self, window, cursor, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIH2x', window, cursor, deviceid))
        return self.send_request(42, buf, is_checked=is_checked)

    def XIChangeHierarchy(self, num_changes, changes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xB3x', num_changes))
        buf.write(xcffib.pack_list(changes, HierarchyChange))
        return self.send_request(43, buf, is_checked=is_checked)

    def XISetClientPointer(self, window, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, deviceid))
        return self.send_request(44, buf, is_checked=is_checked)

    def XIGetClientPointer(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(45, buf, XIGetClientPointerCookie, is_checked=is_checked)

    def XISelectEvents(self, window, num_mask, masks, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, num_mask))
        buf.write(xcffib.pack_list(masks, EventMask))
        return self.send_request(46, buf, is_checked=is_checked)

    def XIQueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', major_version, minor_version))
        return self.send_request(47, buf, XIQueryVersionCookie, is_checked=is_checked)

    def XIQueryDevice(self, deviceid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', deviceid))
        return self.send_request(48, buf, XIQueryDeviceCookie, is_checked=is_checked)

    def XISetFocus(self, window, time, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIH2x', window, time, deviceid))
        return self.send_request(49, buf, is_checked=is_checked)

    def XIGetFocus(self, deviceid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', deviceid))
        return self.send_request(50, buf, XIGetFocusCookie, is_checked=is_checked)

    def XIGrabDevice(self, window, time, cursor, deviceid, mode, paired_device_mode, owner_events, mask_len, mask, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHBBBxH', window, time, cursor, deviceid, mode, paired_device_mode, owner_events, mask_len))
        buf.write(xcffib.pack_list(mask, 'I'))
        return self.send_request(51, buf, XIGrabDeviceCookie, is_checked=is_checked)

    def XIUngrabDevice(self, time, deviceid, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', time, deviceid))
        return self.send_request(52, buf, is_checked=is_checked)

    def XIAllowEvents(self, time, deviceid, event_mode, touchid, grab_window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHBxII', time, deviceid, event_mode, touchid, grab_window))
        return self.send_request(53, buf, is_checked=is_checked)

    def XIPassiveGrabDevice(self, time, grab_window, cursor, detail, deviceid, num_modifiers, mask_len, grab_type, grab_mode, paired_device_mode, owner_events, mask, modifiers, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIHHHBBBB2x', time, grab_window, cursor, detail, deviceid, num_modifiers, mask_len, grab_type, grab_mode, paired_device_mode, owner_events))
        buf.write(xcffib.pack_list(mask, 'I'))
        buf.write(xcffib.pack_list(modifiers, 'I'))
        return self.send_request(54, buf, XIPassiveGrabDeviceCookie, is_checked=is_checked)

    def XIPassiveUngrabDevice(self, grab_window, detail, deviceid, num_modifiers, grab_type, modifiers, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHB3x', grab_window, detail, deviceid, num_modifiers, grab_type))
        buf.write(xcffib.pack_list(modifiers, 'I'))
        return self.send_request(55, buf, is_checked=is_checked)

    def XIListProperties(self, deviceid, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', deviceid))
        return self.send_request(56, buf, XIListPropertiesCookie, is_checked=is_checked)

    def XIChangeProperty(self, deviceid, mode, format, property, type, num_items, items, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHBBIII', deviceid, mode, format, property, type, num_items))
        if format & PropertyFormat._8Bits:
            data8 = items.pop(0)
            items.pop(0)
            buf.write(xcffib.pack_list(data8, 'B'))
            buf.write(struct.pack('=4x'))
        if format & PropertyFormat._16Bits:
            data16 = items.pop(0)
            items.pop(0)
            buf.write(xcffib.pack_list(data16, 'H'))
            buf.write(struct.pack('=4x'))
        if format & PropertyFormat._32Bits:
            data32 = items.pop(0)
            buf.write(xcffib.pack_list(data32, 'I'))
        return self.send_request(57, buf, is_checked=is_checked)

    def XIDeleteProperty(self, deviceid, property, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2xI', deviceid, property))
        return self.send_request(58, buf, is_checked=is_checked)

    def XIGetProperty(self, deviceid, delete, property, type, offset, len, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHBxIIII', deviceid, delete, property, type, offset, len))
        return self.send_request(59, buf, XIGetPropertyCookie, is_checked=is_checked)

    def XIGetSelectedEvents(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(60, buf, XIGetSelectedEventsCookie, is_checked=is_checked)

    def XIBarrierReleasePointer(self, num_barriers, barriers, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', num_barriers))
        buf.write(xcffib.pack_list(barriers, BarrierReleasePointerInfo))
        return self.send_request(61, buf, is_checked=is_checked)

    def SendExtensionEvent(self, destination, device_id, propagate, num_classes, num_events, events, classes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIBBHB3x', destination, device_id, propagate, num_classes, num_events))
        buf.write(xcffib.pack_list(events, EventForSend))
        buf.write(xcffib.pack_list(classes, 'I'))
        return self.send_request(31, buf, is_checked=is_checked)