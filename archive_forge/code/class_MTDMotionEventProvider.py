import os
import os.path
import time
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class MTDMotionEventProvider(MotionEventProvider):
    options = ('min_position_x', 'max_position_x', 'min_position_y', 'max_position_y', 'min_pressure', 'max_pressure', 'min_touch_major', 'max_touch_major', 'min_touch_minor', 'max_touch_minor', 'invert_x', 'invert_y', 'rotation')

    def __init__(self, device, args):
        super(MTDMotionEventProvider, self).__init__(device, args)
        self._device = None
        self.input_fn = None
        self.default_ranges = dict()
        args = args.split(',')
        if not args:
            Logger.error('MTD: No filename pass to MTD configuration')
            Logger.error('MTD: Use /dev/input/event0 for example')
            return
        self.input_fn = args[0]
        Logger.info('MTD: Read event from <%s>' % self.input_fn)
        for arg in args[1:]:
            if arg == '':
                continue
            arg = arg.split('=')
            if len(arg) != 2:
                err = 'MTD: Bad parameter %s: Not in key=value format' % arg
                Logger.error(err)
                continue
            key, value = arg
            if key not in MTDMotionEventProvider.options:
                Logger.error('MTD: unknown %s option' % key)
                continue
            try:
                self.default_ranges[key] = int(value)
            except ValueError:
                err = 'MTD: invalid value %s for option %s' % (key, value)
                Logger.error(err)
                continue
            Logger.info('MTD: Set custom %s to %d' % (key, int(value)))
        if 'rotation' not in self.default_ranges:
            self.default_ranges['rotation'] = 0
        elif self.default_ranges['rotation'] not in (0, 90, 180, 270):
            Logger.error('HIDInput: invalid rotation value ({})'.format(self.default_ranges['rotation']))
            self.default_ranges['rotation'] = 0

    def start(self):
        if self.input_fn is None:
            return
        self.uid = 0
        self.queue = collections.deque()
        self.thread = threading.Thread(name=self.__class__.__name__, target=self._thread_run, kwargs=dict(queue=self.queue, input_fn=self.input_fn, device=self.device, default_ranges=self.default_ranges))
        self.thread.daemon = True
        self.thread.start()

    def _thread_run(self, **kwargs):
        input_fn = kwargs.get('input_fn')
        queue = kwargs.get('queue')
        device = kwargs.get('device')
        drs = kwargs.get('default_ranges').get
        touches = {}
        touches_sent = []
        point = {}
        l_points = {}

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

        def process(points):
            for args in points:
                if 'id' not in args:
                    continue
                tid = args['id']
                try:
                    touch = touches[tid]
                except KeyError:
                    touch = MTDMotionEvent(device, tid, args)
                    touches[touch.id] = touch
                touch.move(args)
                action = 'update'
                if tid not in touches_sent:
                    action = 'begin'
                    touches_sent.append(tid)
                if 'delete' in args:
                    action = 'end'
                    del args['delete']
                    del touches[touch.id]
                    touches_sent.remove(tid)
                    touch.update_time_end()
                queue.append((action, touch))

        def normalize(value, vmin, vmax):
            try:
                return (value - vmin) / float(vmax - vmin)
            except ZeroDivisionError:
                return value - vmin
        _fn = input_fn
        _slot = 0
        try:
            _device = Device(_fn)
        except OSError as e:
            if e.errno == 13:
                Logger.warn('MTD: Unable to open device "{0}". Please ensure you have the appropriate permissions.'.format(_fn))
                return
            else:
                raise
        _changes = set()
        ab = _device.get_abs(MTDEV_ABS_POSITION_X)
        range_min_position_x = drs('min_position_x', ab.minimum)
        range_max_position_x = drs('max_position_x', ab.maximum)
        Logger.info('MTD: <%s> range position X is %d - %d' % (_fn, range_min_position_x, range_max_position_x))
        ab = _device.get_abs(MTDEV_ABS_POSITION_Y)
        range_min_position_y = drs('min_position_y', ab.minimum)
        range_max_position_y = drs('max_position_y', ab.maximum)
        Logger.info('MTD: <%s> range position Y is %d - %d' % (_fn, range_min_position_y, range_max_position_y))
        ab = _device.get_abs(MTDEV_ABS_TOUCH_MAJOR)
        range_min_major = drs('min_touch_major', ab.minimum)
        range_max_major = drs('max_touch_major', ab.maximum)
        Logger.info('MTD: <%s> range touch major is %d - %d' % (_fn, range_min_major, range_max_major))
        ab = _device.get_abs(MTDEV_ABS_TOUCH_MINOR)
        range_min_minor = drs('min_touch_minor', ab.minimum)
        range_max_minor = drs('max_touch_minor', ab.maximum)
        Logger.info('MTD: <%s> range touch minor is %d - %d' % (_fn, range_min_minor, range_max_minor))
        range_min_pressure = drs('min_pressure', 0)
        range_max_pressure = drs('max_pressure', 255)
        Logger.info('MTD: <%s> range pressure is %d - %d' % (_fn, range_min_pressure, range_max_pressure))
        invert_x = int(bool(drs('invert_x', 0)))
        invert_y = int(bool(drs('invert_y', 0)))
        Logger.info('MTD: <%s> axes inversion: X is %d, Y is %d' % (_fn, invert_x, invert_y))
        rotation = drs('rotation', 0)
        Logger.info('MTD: <%s> rotation set to %d' % (_fn, rotation))
        failures = 0
        while _device:
            if failures > 1000:
                Logger.info('MTD: <%s> input device disconnected' % _fn)
                while not os.path.exists(_fn):
                    time.sleep(0.05)
                _device.close()
                _device = Device(_fn)
                Logger.info('MTD: <%s> input device reconnected' % _fn)
                failures = 0
                continue
            while _device.idle(1000):
                continue
            while True:
                data = _device.get()
                if data is None:
                    failures += 1
                    break
                failures = 0
                if data.type == MTDEV_TYPE_EV_ABS and data.code == MTDEV_CODE_SLOT:
                    _slot = data.value
                    continue
                if not _slot in l_points:
                    l_points[_slot] = dict()
                point = l_points[_slot]
                ev_value = data.value
                ev_code = data.code
                if ev_code == MTDEV_CODE_POSITION_X:
                    val = normalize(ev_value, range_min_position_x, range_max_position_x)
                    assign_coord(point, val, invert_x, 'xy')
                elif ev_code == MTDEV_CODE_POSITION_Y:
                    val = 1.0 - normalize(ev_value, range_min_position_y, range_max_position_y)
                    assign_coord(point, val, invert_y, 'yx')
                elif ev_code == MTDEV_CODE_PRESSURE:
                    point['pressure'] = normalize(ev_value, range_min_pressure, range_max_pressure)
                elif ev_code == MTDEV_CODE_TOUCH_MAJOR:
                    point['size_w'] = normalize(ev_value, range_min_major, range_max_major)
                elif ev_code == MTDEV_CODE_TOUCH_MINOR:
                    point['size_h'] = normalize(ev_value, range_min_minor, range_max_minor)
                elif ev_code == MTDEV_CODE_TRACKING_ID:
                    if ev_value == -1:
                        point['delete'] = True
                        _changes.add(_slot)
                        process([l_points[x] for x in _changes])
                        _changes.clear()
                        continue
                    else:
                        point['id'] = ev_value
                else:
                    continue
                _changes.add(_slot)
            if _changes:
                process([l_points[x] for x in _changes])
                _changes.clear()

    def update(self, dispatch_fn):
        try:
            while True:
                event_type, touch = self.queue.popleft()
                dispatch_fn(event_type, touch)
        except:
            pass