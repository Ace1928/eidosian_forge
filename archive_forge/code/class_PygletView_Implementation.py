from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
class PygletView_Implementation:
    PygletView = cocoapy.ObjCSubclass('NSView', 'PygletView')

    @PygletView.method(b'@' + cocoapy.NSRectEncoding + cocoapy.PyObjectEncoding)
    def initWithFrame_cocoaWindow_(self, frame, window):
        self._tracking_area = None
        self = cocoapy.ObjCInstance(cocoapy.send_super(self, 'initWithFrame:', frame, argtypes=[cocoapy.NSRect]))
        if not self:
            return None
        self._window = window
        self.updateTrackingAreas()
        self._textview = PygletTextView.alloc().initWithCocoaWindow_(window)
        self.addSubview_(self._textview)
        return self

    @PygletView.method('v')
    def dealloc(self):
        self._window = None
        self._textview.release()
        self._textview = None
        self._tracking_area.release()
        self._tracking_area = None
        cocoapy.send_super(self, 'dealloc')

    @PygletView.method('v')
    def updateTrackingAreas(self):
        if self._tracking_area:
            self.removeTrackingArea_(self._tracking_area)
            self._tracking_area.release()
            self._tracking_area = None
        tracking_options = cocoapy.NSTrackingMouseEnteredAndExited | cocoapy.NSTrackingActiveInActiveApp | cocoapy.NSTrackingCursorUpdate
        frame = self.frame()
        self._tracking_area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(frame, tracking_options, self, None)
        self.addTrackingArea_(self._tracking_area)

    @PygletView.method('B')
    def canBecomeKeyView(self):
        return True

    @PygletView.method('B')
    def isOpaque(self):
        return True

    @PygletView.method(b'v' + cocoapy.NSSizeEncoding)
    def setFrameSize_(self, size):
        cocoapy.send_super(self, 'setFrameSize:', size, superclass_name='NSView', argtypes=[cocoapy.NSSize])
        if not self._window.context.canvas:
            return
        width, height = (int(size.width), int(size.height))
        self._window.switch_to()
        self._window.context.update_geometry()
        self._window._width, self._window._height = (width, height)
        self._window.dispatch_event('on_resize', width, height)
        self._window.dispatch_event('on_expose')
        if self.inLiveResize():
            from pyglet import app
            if app.event_loop is not None:
                app.event_loop.idle()

    @PygletView.method('v@')
    def keyDown_(self, nsevent):
        if not nsevent.isARepeat():
            symbol = getSymbol(nsevent)
            modifiers = getModifiers(nsevent)
            self._window.dispatch_event('on_key_press', symbol, modifiers)

    @PygletView.method('v@')
    def keyUp_(self, nsevent):
        symbol = getSymbol(nsevent)
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_key_release', symbol, modifiers)

    @PygletView.method('v@')
    def flagsChanged_(self, nsevent):
        symbol = keymap.get(nsevent.keyCode(), None)
        if symbol is None or symbol not in maskForKey:
            return
        modifiers = getModifiers(nsevent)
        modifierFlags = nsevent.modifierFlags()
        if symbol and modifierFlags & maskForKey[symbol]:
            self._window.dispatch_event('on_key_press', symbol, modifiers)
        else:
            self._window.dispatch_event('on_key_release', symbol, modifiers)

    @PygletView.method('B@')
    def performKeyEquivalent_(self, nsevent):
        modifierFlags = nsevent.modifierFlags()
        if modifierFlags & cocoapy.NSNumericPadKeyMask:
            return False
        if modifierFlags & cocoapy.NSFunctionKeyMask:
            ch = cocoapy.cfstring_to_string(nsevent.charactersIgnoringModifiers())
            if ch in (cocoapy.NSHomeFunctionKey, cocoapy.NSEndFunctionKey, cocoapy.NSPageUpFunctionKey, cocoapy.NSPageDownFunctionKey):
                return False
        NSApp = cocoapy.ObjCClass('NSApplication').sharedApplication()
        NSApp.mainMenu().performKeyEquivalent_(nsevent)
        return True

    @PygletView.method('v@')
    def mouseMoved_(self, nsevent):
        if self._window._mouse_ignore_motion:
            self._window._mouse_ignore_motion = False
            return
        if not self._window._mouse_in_window:
            return
        x, y = getMousePosition(self, nsevent)
        dx, dy = getMouseDelta(nsevent)
        self._window.dispatch_event('on_mouse_motion', x, y, dx, dy)

    @PygletView.method('v@')
    def scrollWheel_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        scroll_x, scroll_y = getMouseDelta(nsevent)
        self._window.dispatch_event('on_mouse_scroll', x, y, scroll_x, scroll_y)

    @PygletView.method('v@')
    def mouseDown_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.LEFT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_press', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def mouseDragged_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        dx, dy = getMouseDelta(nsevent)
        buttons = mouse.LEFT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)

    @PygletView.method('v@')
    def mouseUp_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.LEFT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_release', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def rightMouseDown_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.RIGHT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_press', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def rightMouseDragged_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        dx, dy = getMouseDelta(nsevent)
        buttons = mouse.RIGHT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)

    @PygletView.method('v@')
    def rightMouseUp_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.RIGHT
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_release', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def otherMouseDown_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.MIDDLE
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_press', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def otherMouseDragged_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        dx, dy = getMouseDelta(nsevent)
        buttons = mouse.MIDDLE
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)

    @PygletView.method('v@')
    def otherMouseUp_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        buttons = mouse.MIDDLE
        modifiers = getModifiers(nsevent)
        self._window.dispatch_event('on_mouse_release', x, y, buttons, modifiers)

    @PygletView.method('v@')
    def mouseEntered_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        self._window._mouse_in_window = True
        self._window.dispatch_event('on_mouse_enter', x, y)

    @PygletView.method('v@')
    def mouseExited_(self, nsevent):
        x, y = getMousePosition(self, nsevent)
        self._window._mouse_in_window = False
        if not self._window._mouse_exclusive:
            self._window.set_mouse_platform_visible()
        self._window.dispatch_event('on_mouse_leave', x, y)

    @PygletView.method('v@')
    def cursorUpdate_(self, nsevent):
        self._window._mouse_in_window = True
        if not self._window._mouse_exclusive:
            self._window.set_mouse_platform_visible()

    @PygletView.method('Q@')
    def draggingEntered_(self, draginfo):
        return cocoapy.NSDragOperationGeneric

    @PygletView.method('B@')
    def performDragOperation_(self, sender):
        pos = sender.draggingLocation()
        pasteboard = sender.draggingPasteboard()
        classes = NSArray.arrayWithObject_(NSURL)
        options = NSDictionary.dictionaryWithObject_forKey_(NSNumber.numberWithBool_(True), NSPasteboardURLReadingFileURLsOnlyKey)
        urls = pasteboard.readObjectsForClasses_options_(classes, options)
        url_count = urls.count()
        paths = []
        for i in range(url_count):
            fpath = urls.objectAtIndex_(i).fileSystemRepresentation()
            paths.append(fpath.decode())
        self._window.dispatch_event('on_file_drop', pos.x, pos.y, paths)