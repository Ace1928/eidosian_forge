from ctypes import c_void_p, c_bool
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, send_super
from pyglet.libs.darwin.cocoapy import NSUInteger, NSUIntegerEncoding
from pyglet.libs.darwin.cocoapy import NSRectEncoding
class PygletWindow_Implementation:
    PygletWindow = ObjCSubclass('NSWindow', 'PygletWindow')

    @PygletWindow.method('B')
    def canBecomeKeyWindow(self):
        return True

    @PygletWindow.method(b'@' + NSUIntegerEncoding + b'@@B')
    def nextEventMatchingMask_untilDate_inMode_dequeue_(self, mask, date, mode, dequeue):
        if self.inLiveResize():
            from pyglet import app
            if app.event_loop is not None:
                app.event_loop.idle()
        event = send_super(self, 'nextEventMatchingMask:untilDate:inMode:dequeue:', mask, date, mode, dequeue, superclass_name='NSWindow', argtypes=[NSUInteger, c_void_p, c_void_p, c_bool])
        if event.value is None:
            return 0
        else:
            return event.value

    @PygletWindow.method(b'd' + NSRectEncoding)
    def animationResizeTime_(self, newFrame):
        return 0.0