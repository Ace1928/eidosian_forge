from kivy.config import Config
from kivy.logger import Logger
from kivy.input import providers
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
class InputPostprocCalibration(object):
    """Recalibrate the inputs.

    The configuration must go within a section named `postproc:calibration`.
    Within the section, you must have a line like::

        devicename = param=value,param=value

    If you wish to match by provider, you must have a line like::

        (provider) = param=value,param=value

    :Parameters:
        `xratio`: float
            Value to multiply X
        `yratio`: float
            Value to multiply Y
        `xoffset`: float
            Value to add to X
        `yoffset`: float
            Value to add to Y
        `auto`: str
            If set, then the touch is transformed from screen-relative
            to window-relative The value is used as an indication of
            screen size, e.g for fullHD:

                auto=1920x1080

            If present, this setting overrides all the others.
            This assumes the input device exactly covers the display
            area, if they are different, the computations will be wrong.

    .. versionchanged:: 1.11.0
        Added `auto` parameter
    """

    def __init__(self):
        super(InputPostprocCalibration, self).__init__()
        self.devices = {}
        self.frame = 0
        self.provider_map = self._get_provider_map()
        if not Config.has_section('postproc:calibration'):
            return
        default_params = {'xoffset': 0, 'yoffset': 0, 'xratio': 1, 'yratio': 1}
        for device_key, params_str in Config.items('postproc:calibration'):
            params = default_params.copy()
            for param in params_str.split(','):
                param = param.strip()
                if not param:
                    continue
                key, value = param.split('=', 1)
                if key == 'auto':
                    width, height = [float(x) for x in value.split('x')]
                    params['auto'] = (width, height)
                    break
                if key not in ('xoffset', 'yoffset', 'xratio', 'yratio'):
                    Logger.error('Calibration: invalid key provided: {}'.format(key))
                params[key] = float(value)
            self.devices[device_key] = params

    def _get_provider_map(self):
        """Iterates through all registered input provider names and finds the
        respective MotionEvent subclass for each. Returns a dict of MotionEvent
        subclasses mapped to their provider name.
        """
        provider_map = {}
        for input_provider in MotionEventFactory.list():
            if not hasattr(providers, input_provider):
                continue
            p = getattr(providers, input_provider)
            for m in p.__all__:
                event = getattr(p, m)
                if issubclass(event, MotionEvent):
                    provider_map[event] = input_provider
        return provider_map

    def _get_provider_key(self, event):
        """Returns the provider key for the event if the provider is configured
        for calibration.
        """
        input_type = self.provider_map.get(event.__class__)
        key = '({})'.format(input_type)
        if input_type and key in self.devices:
            return key

    def process(self, events):
        if not self.devices:
            return events
        self.frame += 1
        frame = self.frame
        to_remove = []
        for etype, event in events:
            if etype == 'end':
                continue
            if event.device in self.devices:
                dev = event.device
            else:
                dev = self._get_provider_key(event)
            if not dev:
                continue
            if 'calibration:frame' not in event.ud:
                event.ud['calibration:frame'] = frame
            elif event.ud['calibration:frame'] == frame:
                continue
            event.ud['calibration:frame'] = frame
            params = self.devices[dev]
            if 'auto' in params:
                event.sx, event.sy = self.auto_calibrate(event.sx, event.sy, params['auto'])
                if not (0 <= event.sx <= 1 and 0 <= event.sy <= 1):
                    to_remove.append((etype, event))
            else:
                event.sx = event.sx * params['xratio'] + params['xoffset']
                event.sy = event.sy * params['yratio'] + params['yoffset']
        for event in to_remove:
            events.remove(event)
        return events

    def auto_calibrate(self, sx, sy, size):
        from kivy.core.window import Window as W
        WIDTH, HEIGHT = size
        xratio = WIDTH / W.width
        yratio = HEIGHT / W.height
        xoffset = -W.left / W.width
        yoffset = -(HEIGHT - W.top - W.height) / W.height
        sx = sx * xratio + xoffset
        sy = sy * yratio + yoffset
        return (sx, sy)