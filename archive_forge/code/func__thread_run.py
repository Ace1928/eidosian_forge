import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def _thread_run(self, **kwargs):
    input_fn = kwargs.get('input_fn')
    queue = self.queue
    dispatch_queue = self.dispatch_queue
    device = kwargs.get('device')
    drs = kwargs.get('default_ranges').get
    touches = {}
    touches_sent = []
    point = {}
    l_points = []
    range_min_position_x = 0
    range_max_position_x = 2048
    range_min_position_y = 0
    range_max_position_y = 2048
    range_min_pressure = 0
    range_max_pressure = 255
    range_min_abs_x = 0
    range_max_abs_x = 255
    range_min_abs_y = 0
    range_max_abs_y = 255
    range_min_abs_pressure = 0
    range_max_abs_pressure = 255
    invert_x = int(bool(drs('invert_x', 0)))
    invert_y = int(bool(drs('invert_y', 1)))
    rotation = drs('rotation', 0)

    def assign_coord(point, value, invert, coords):
        cx, cy = coords
        if invert:
            value = 1.0 - value
        if rotation == 0:
            point[cx] = value
        elif rotation == 90:
            point[cy] = value
        elif rotation == 180:
            point[cx] = 1.0 - value
        elif rotation == 270:
            point[cy] = 1.0 - value

    def assign_rel_coord(point, value, invert, coords):
        cx, cy = coords
        if invert:
            value = -1 * value
        if rotation == 0:
            point[cx] += value
        elif rotation == 90:
            point[cy] += value
        elif rotation == 180:
            point[cx] += -value
        elif rotation == 270:
            point[cy] += -value
        point['x'] = min(1.0, max(0.0, point['x']))
        point['y'] = min(1.0, max(0.0, point['y']))

    def process_as_multitouch(tv_sec, tv_usec, ev_type, ev_code, ev_value):
        if ev_type == EV_SYN:
            if ev_code == SYN_MT_REPORT:
                if 'id' not in point:
                    return
                l_points.append(point.copy())
            elif ev_code == SYN_REPORT:
                process(l_points)
                del l_points[:]
        elif ev_type == EV_MSC and ev_code in (MSC_RAW, MSC_SCAN):
            pass
        elif ev_code == ABS_MT_TRACKING_ID:
            point.clear()
            point['id'] = ev_value
        elif ev_code == ABS_MT_POSITION_X:
            val = normalize(ev_value, range_min_position_x, range_max_position_x)
            assign_coord(point, val, invert_x, 'xy')
        elif ev_code == ABS_MT_POSITION_Y:
            val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
            assign_coord(point, val, invert_y, 'yx')
        elif ev_code == ABS_MT_ORIENTATION:
            point['orientation'] = ev_value
        elif ev_code == ABS_MT_BLOB_ID:
            point['blobid'] = ev_value
        elif ev_code == ABS_MT_PRESSURE:
            point['pressure'] = normalize(ev_value, range_min_pressure, range_max_pressure)
        elif ev_code == ABS_MT_TOUCH_MAJOR:
            point['size_w'] = ev_value
        elif ev_code == ABS_MT_TOUCH_MINOR:
            point['size_h'] = ev_value

    def process_as_mouse_or_keyboard(tv_sec, tv_usec, ev_type, ev_code, ev_value):
        if ev_type == EV_SYN:
            if ev_code == SYN_REPORT:
                process([point])
                if 'button' in point and point['button'].startswith('scroll'):
                    del point['button']
                    point['id'] += 1
                    point['_avoid'] = True
                    process([point])
        elif ev_type == EV_REL:
            if ev_code == 0:
                assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_x, 'xy')
            elif ev_code == 1:
                assign_rel_coord(point, min(1.0, max(-1.0, ev_value / 1000.0)), invert_y, 'yx')
            elif ev_code == 8:
                b = 'scrollup' if ev_value < 0 else 'scrolldown'
                if 'button' not in point:
                    point['button'] = b
                    point['id'] += 1
                    if '_avoid' in point:
                        del point['_avoid']
        elif ev_type != EV_KEY:
            if ev_code == ABS_X:
                val = normalize(ev_value, range_min_abs_x, range_max_abs_x)
                assign_coord(point, val, invert_x, 'xy')
            elif ev_code == ABS_Y:
                val = 1.0 - normalize(ev_value, range_min_abs_y, range_max_abs_y)
                assign_coord(point, val, invert_y, 'yx')
            elif ev_code == ABS_PRESSURE:
                point['pressure'] = normalize(ev_value, range_min_abs_pressure, range_max_abs_pressure)
        else:
            buttons = {272: 'left', 273: 'right', 274: 'middle', 275: 'side', 276: 'extra', 277: 'forward', 278: 'back', 279: 'task', 330: 'touch', 320: 'pen'}
            if ev_code in buttons.keys():
                if ev_value:
                    if 'button' not in point:
                        point['button'] = buttons[ev_code]
                        point['id'] += 1
                        if '_avoid' in point:
                            del point['_avoid']
                elif 'button' in point:
                    if point['button'] == buttons[ev_code]:
                        del point['button']
                        point['id'] += 1
                        point['_avoid'] = True
            else:
                if not 0 <= ev_value <= 1:
                    return
                if ev_code not in keyboard_keys:
                    Logger.warn('HIDInput: unhandled HID code: {}'.format(ev_code))
                    return
                z = keyboard_keys[ev_code][-1 if 'shift' in Window._modifiers else 0]
                if z.lower() not in Keyboard.keycodes:
                    Logger.warn('HIDInput: unhandled character: {}'.format(z))
                    return
                keycode = Keyboard.keycodes[z.lower()]
                if ev_value == 1:
                    if z == 'shift' or z == 'alt':
                        Window._modifiers.append(z)
                    elif z.endswith('ctrl'):
                        Window._modifiers.append('ctrl')
                    dispatch_queue.append(('key_down', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
                elif ev_value == 0:
                    dispatch_queue.append(('key_up', (keycode, ev_code, keys_str.get(z, z), Window._modifiers)))
                    if (z == 'shift' or z == 'alt') and z in Window._modifiers:
                        Window._modifiers.remove(z)
                    elif z.endswith('ctrl') and 'ctrl' in Window._modifiers:
                        Window._modifiers.remove('ctrl')

    def process(points):
        if not is_multitouch:
            dispatch_queue.append(('mouse_pos', (points[0]['x'] * Window.width, points[0]['y'] * Window.height)))
        actives = [args['id'] for args in points if 'id' in args and '_avoid' not in args]
        for args in points:
            tid = args['id']
            try:
                touch = touches[tid]
                if touch.sx == args['x'] and touch.sy == args['y']:
                    continue
                touch.move(args)
                if tid not in touches_sent:
                    queue.append(('begin', touch))
                    touches_sent.append(tid)
                queue.append(('update', touch))
            except KeyError:
                if '_avoid' not in args:
                    touch = HIDMotionEvent(device, tid, args)
                    touches[touch.id] = touch
                    if tid not in touches_sent:
                        queue.append(('begin', touch))
                        touches_sent.append(tid)
        for tid in list(touches.keys())[:]:
            if tid not in actives:
                touch = touches[tid]
                if tid in touches_sent:
                    touch.update_time_end()
                    queue.append(('end', touch))
                    touches_sent.remove(tid)
                del touches[tid]

    def normalize(value, vmin, vmax):
        return (value - vmin) / float(vmax - vmin)
    fd = open(input_fn, 'rb')
    device_name = fcntl.ioctl(fd, EVIOCGNAME + (256 << 16), ' ' * 256).decode().strip()
    Logger.info('HIDMotionEvent: using <%s>' % device_name)
    bit = fcntl.ioctl(fd, EVIOCGBIT + (EV_MAX << 16), ' ' * sz_l)
    bit, = struct.unpack('Q', bit)
    is_multitouch = False
    for x in range(EV_MAX):
        if x != EV_ABS:
            continue
        if bit & 1 << x == 0:
            continue
        sbit = fcntl.ioctl(fd, EVIOCGBIT + x + (KEY_MAX << 16), ' ' * sz_l)
        sbit, = struct.unpack('Q', sbit)
        for y in range(KEY_MAX):
            if sbit & 1 << y == 0:
                continue
            absinfo = fcntl.ioctl(fd, EVIOCGABS + y + (struct_input_absinfo_sz << 16), ' ' * struct_input_absinfo_sz)
            abs_value, abs_min, abs_max, abs_fuzz, abs_flat, abs_res = struct.unpack('iiiiii', absinfo)
            if y == ABS_MT_POSITION_X:
                is_multitouch = True
                range_min_position_x = drs('min_position_x', abs_min)
                range_max_position_x = drs('max_position_x', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range position X is %d - %d' % (device_name, abs_min, abs_max))
            elif y == ABS_MT_POSITION_Y:
                is_multitouch = True
                range_min_position_y = drs('min_position_y', abs_min)
                range_max_position_y = drs('max_position_y', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range position Y is %d - %d' % (device_name, abs_min, abs_max))
            elif y == ABS_MT_PRESSURE:
                range_min_pressure = drs('min_pressure', abs_min)
                range_max_pressure = drs('max_pressure', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range pressure is %d - %d' % (device_name, abs_min, abs_max))
            elif y == ABS_X:
                range_min_abs_x = drs('min_abs_x', abs_min)
                range_max_abs_x = drs('max_abs_x', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range ABS X position is %d - %d' % (device_name, abs_min, abs_max))
            elif y == ABS_Y:
                range_min_abs_y = drs('min_abs_y', abs_min)
                range_max_abs_y = drs('max_abs_y', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range ABS Y position is %d - %d' % (device_name, abs_min, abs_max))
            elif y == ABS_PRESSURE:
                range_min_abs_pressure = drs('min_abs_pressure', abs_min)
                range_max_abs_pressure = drs('max_abs_pressure', abs_max)
                Logger.info('HIDMotionEvent: ' + '<%s> range ABS pressure is %d - %d' % (device_name, abs_min, abs_max))
    if not is_multitouch:
        point = {'x': 0.5, 'y': 0.5, 'id': 0, '_avoid': True}
    while fd:
        data = fd.read(struct_input_event_sz)
        if len(data) < struct_input_event_sz:
            break
        for i in range(int(len(data) / struct_input_event_sz)):
            ev = data[i * struct_input_event_sz:]
            infos = struct.unpack('LLHHi', ev[:struct_input_event_sz])
            if is_multitouch:
                process_as_multitouch(*infos)
            else:
                process_as_mouse_or_keyboard(*infos)