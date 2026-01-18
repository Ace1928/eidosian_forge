import os
from os.path import sep
class ProbeSysfsHardwareProbe(MotionEventProvider):

    def __new__(self, device, args):
        instance = super(ProbeSysfsHardwareProbe, self).__new__(self)
        instance.__init__(device, args)

    def __init__(self, device, args):
        super(ProbeSysfsHardwareProbe, self).__init__(device, args)
        self.provider = 'mtdev'
        self.match = None
        self.input_path = '/sys/class/input'
        self.select_all = True if _is_rpi else False
        self.use_mouse = False
        self.use_regex = False
        self.args = []
        args = args.split(',')
        for arg in args:
            if arg == '':
                continue
            arg = arg.split('=', 1)
            if len(arg) != 2:
                Logger.error('ProbeSysfs: invalid parameters %s, not key=value format' % arg)
                continue
            key, value = arg
            if key == 'match':
                self.match = value
            elif key == 'provider':
                self.provider = value
            elif key == 'use_regex':
                self.use_regex = bool(int(value))
            elif key == 'select_all':
                self.select_all = bool(int(value))
            elif key == 'use_mouse':
                self.use_mouse = bool(int(value))
            elif key == 'param':
                self.args.append(value)
            else:
                Logger.error('ProbeSysfs: unknown %s option' % key)
                continue
        self.probe()

    def should_use_mouse(self):
        return self.use_mouse or not any((p for p in EventLoop.input_providers if isinstance(p, MouseMotionEventProvider)))

    def probe(self):
        global EventLoop
        from kivy.base import EventLoop
        inputs = get_inputs(self.input_path)
        Logger.debug('ProbeSysfs: using probesysfs!')
        use_mouse = self.should_use_mouse()
        if not self.select_all:
            inputs = [x for x in inputs if x.has_capability(ABS_MT_POSITION_X) and (use_mouse or not x.is_mouse)]
        for device in inputs:
            Logger.debug('ProbeSysfs: found device: %s at %s' % (device.name, device.device))
            if self.match:
                if self.use_regex:
                    if not match(self.match, device.name, IGNORECASE):
                        Logger.debug('ProbeSysfs: device not match the rule in config, ignoring.')
                        continue
                elif self.match not in device.name:
                    continue
            Logger.info('ProbeSysfs: device match: %s' % device.device)
            d = device.device
            devicename = self.device % dict(name=d.split(sep)[-1])
            provider = MotionEventFactory.get(self.provider)
            if provider is None:
                Logger.info('ProbeSysfs: Unable to find provider %s' % self.provider)
                Logger.info('ProbeSysfs: fallback on hidinput')
                provider = MotionEventFactory.get('hidinput')
            if provider is None:
                Logger.critical('ProbeSysfs: no input provider found to handle this device !')
                continue
            instance = provider(devicename, '%s,%s' % (device.device, ','.join(self.args)))
            if instance:
                EventLoop.add_input_provider(instance)