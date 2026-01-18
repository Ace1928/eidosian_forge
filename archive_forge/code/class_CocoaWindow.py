from ctypes import *
import pyglet
from pyglet.window import BaseWindow
from pyglet.window import MouseCursor, DefaultMouseCursor
from pyglet.window import WindowException
from pyglet.event import EventDispatcher
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, CGPoint, AutoReleasePool
from .systemcursor import SystemCursor
from .pyglet_delegate import PygletDelegate
from .pyglet_window import PygletWindow, PygletToolWindow
from .pyglet_view import PygletView
class CocoaWindow(BaseWindow):

    def __init__(self, width=None, height=None, caption=None, resizable=False, style=BaseWindow.WINDOW_STYLE_DEFAULT, fullscreen=False, visible=True, vsync=True, file_drops=False, display=None, screen=None, config=None, context=None, mode=None):
        with AutoReleasePool():
            super().__init__(width, height, caption, resizable, style, fullscreen, visible, vsync, file_drops, display, screen, config, context, mode)
    _nswindow = None
    _delegate = None
    _mouse_platform_visible = True
    _mouse_ignore_motion = False
    _was_closed = False
    _style_masks = {BaseWindow.WINDOW_STYLE_DEFAULT: cocoapy.NSTitledWindowMask | cocoapy.NSClosableWindowMask | cocoapy.NSMiniaturizableWindowMask, BaseWindow.WINDOW_STYLE_DIALOG: cocoapy.NSTitledWindowMask | cocoapy.NSClosableWindowMask, BaseWindow.WINDOW_STYLE_TOOL: cocoapy.NSTitledWindowMask | cocoapy.NSClosableWindowMask | cocoapy.NSUtilityWindowMask, BaseWindow.WINDOW_STYLE_BORDERLESS: cocoapy.NSBorderlessWindowMask}

    def _recreate(self, changes):
        if 'context' in changes:
            self.context.set_current()
        if 'fullscreen' in changes:
            if not self._fullscreen:
                self.screen.release_display()
        self._create()

    def _create(self):
        with AutoReleasePool():
            if self._nswindow:
                nsview = self.canvas.nsview
                self.canvas = None
                self._nswindow.orderOut_(None)
                self._nswindow.close()
                self.context.detach()
                self._nswindow.release()
                self._nswindow = None
                nsview.release()
                self._delegate.release()
                self._delegate = None
            content_rect = cocoapy.NSMakeRect(0, 0, self._width, self._height)
            WindowClass = PygletWindow
            if self._fullscreen:
                style_mask = cocoapy.NSBorderlessWindowMask
            else:
                if self._style not in self._style_masks:
                    self._style = self.WINDOW_STYLE_DEFAULT
                style_mask = self._style_masks[self._style]
                if self._resizable:
                    style_mask |= cocoapy.NSResizableWindowMask
                if self._style == BaseWindow.WINDOW_STYLE_TOOL:
                    WindowClass = PygletToolWindow
            self._nswindow = WindowClass.alloc().initWithContentRect_styleMask_backing_defer_(content_rect, style_mask, cocoapy.NSBackingStoreBuffered, False)
            if self._fullscreen:
                blackColor = NSColor.blackColor()
                self._nswindow.setBackgroundColor_(blackColor)
                self._nswindow.setOpaque_(True)
                self.screen.capture_display()
                self._nswindow.setLevel_(quartz.CGShieldingWindowLevel())
                self.context.set_full_screen()
                self._center_window()
                self._mouse_in_window = True
            else:
                self._set_nice_window_location()
                self._mouse_in_window = self._mouse_in_content_rect()
            self._nsview = PygletView.alloc().initWithFrame_cocoaWindow_(content_rect, self)
            self._nswindow.setContentView_(self._nsview)
            self._nswindow.makeFirstResponder_(self._nsview)
            self.canvas = CocoaCanvas(self.display, self.screen, self._nsview)
            self.context.attach(self.canvas)
            self._nswindow.setAcceptsMouseMovedEvents_(True)
            self._nswindow.setReleasedWhenClosed_(False)
            self._nswindow.useOptimizedDrawing_(True)
            self._nswindow.setPreservesContentDuringLiveResize_(False)
            self._delegate = PygletDelegate.alloc().initWithWindow_(self)
            self.set_caption(self._caption)
            if self._minimum_size is not None:
                self.set_minimum_size(*self._minimum_size)
            if self._maximum_size is not None:
                self.set_maximum_size(*self._maximum_size)
            if self._file_drops:
                array = NSArray.arrayWithObject_(cocoapy.NSPasteboardTypeURL)
                self._nsview.registerForDraggedTypes_(array)
            self.context.update_geometry()
            self.switch_to()
            self.set_vsync(self._vsync)
            self.set_visible(self._visible)

    def _set_nice_window_location(self):
        visible_windows = [win for win in pyglet.app.windows if win is not self and win._nswindow and win._nswindow.isVisible()]
        if not visible_windows:
            self._center_window()
        else:
            point = visible_windows[-1]._nswindow.cascadeTopLeftFromPoint_(cocoapy.NSZeroPoint)
            self._nswindow.cascadeTopLeftFromPoint_(point)

    def _center_window(self):
        x = self.screen.x + int((self.screen.width - self._width) // 2)
        y = self.screen.y + int((self.screen.height - self._height) // 2)
        self._nswindow.setFrameOrigin_(cocoapy.NSPoint(x, y))

    def close(self):
        if self._was_closed:
            return
        with AutoReleasePool():
            self.set_mouse_platform_visible(True)
            self.set_exclusive_mouse(False)
            self.set_exclusive_keyboard(False)
            if self._delegate:
                self._nswindow.setDelegate_(None)
                self._delegate.release()
                self._delegate = None
            if self._nswindow:
                self._nswindow.orderOut_(None)
                self._nswindow.setContentView_(None)
                self._nswindow.close()
            self.screen.restore_mode()
            if self.canvas:
                self.canvas.nsview.release()
                self.canvas.nsview = None
                self.canvas = None
            super(CocoaWindow, self).close()
            self._was_closed = True

    def switch_to(self):
        if self.context:
            self.context.set_current()

    def flip(self):
        self.draw_mouse_cursor()
        if self.context:
            self.context.flip()

    def dispatch_events(self):
        self._allow_dispatch_event = True
        self.dispatch_pending_events()
        event = True
        with AutoReleasePool():
            NSApp = NSApplication.sharedApplication()
            while event and self._nswindow and self._context:
                event = NSApp.nextEventMatchingMask_untilDate_inMode_dequeue_(cocoapy.NSAnyEventMask, None, cocoapy.NSEventTrackingRunLoopMode, True)
                if event:
                    event_type = event.type()
                    NSApp.sendEvent_(event)
                    if event_type == cocoapy.NSKeyDown and (not event.isARepeat()):
                        NSApp.sendAction_to_from_(cocoapy.get_selector('pygletKeyDown:'), None, event)
                    elif event_type == cocoapy.NSKeyUp:
                        NSApp.sendAction_to_from_(cocoapy.get_selector('pygletKeyUp:'), None, event)
                    elif event_type == cocoapy.NSFlagsChanged:
                        NSApp.sendAction_to_from_(cocoapy.get_selector('pygletFlagsChanged:'), None, event)
                    NSApp.updateWindows()
        self._allow_dispatch_event = False

    def dispatch_pending_events(self):
        while self._event_queue:
            event = self._event_queue.pop(0)
            EventDispatcher.dispatch_event(self, *event)

    def set_caption(self, caption):
        self._caption = caption
        if self._nswindow is not None:
            self._nswindow.setTitle_(cocoapy.get_NSString(caption))

    def set_icon(self, *images):
        max_image = images[0]
        for img in images:
            if img.width > max_image.width and img.height > max_image.height:
                max_image = img
        image = max_image.get_image_data()
        format = 'ARGB'
        bytesPerRow = len(format) * image.width
        data = image.get_data(format, -bytesPerRow)
        cfdata = c_void_p(cf.CFDataCreate(None, data, len(data)))
        provider = c_void_p(quartz.CGDataProviderCreateWithCFData(cfdata))
        colorSpace = c_void_p(quartz.CGColorSpaceCreateDeviceRGB())
        cgimage = c_void_p(quartz.CGImageCreate(image.width, image.height, 8, 32, bytesPerRow, colorSpace, cocoapy.kCGImageAlphaFirst, provider, None, True, cocoapy.kCGRenderingIntentDefault))
        if not cgimage:
            return
        cf.CFRelease(cfdata)
        quartz.CGDataProviderRelease(provider)
        quartz.CGColorSpaceRelease(colorSpace)
        size = cocoapy.NSMakeSize(image.width, image.height)
        nsimage = NSImage.alloc().initWithCGImage_size_(cgimage, size)
        if not nsimage:
            return
        NSApp = NSApplication.sharedApplication()
        NSApp.setApplicationIconImage_(nsimage)
        nsimage.release()

    def get_location(self):
        window_frame = self._nswindow.frame()
        rect = self._nswindow.contentRectForFrameRect_(window_frame)
        screen_frame = self._nswindow.screen().frame()
        screen_width = int(screen_frame.size.width)
        screen_height = int(screen_frame.size.height)
        return (int(rect.origin.x), int(screen_height - rect.origin.y - rect.size.height))

    def set_location(self, x, y):
        window_frame = self._nswindow.frame()
        rect = self._nswindow.contentRectForFrameRect_(window_frame)
        screen_frame = self._nswindow.screen().frame()
        screen_width = int(screen_frame.size.width)
        screen_height = int(screen_frame.size.height)
        origin = cocoapy.NSPoint(x, screen_height - y - rect.size.height)
        self._nswindow.setFrameOrigin_(origin)

    def get_framebuffer_size(self):
        view = self.context._nscontext.view()
        bounds = view.convertRectToBacking_(view.bounds()).size
        return (int(bounds.width), int(bounds.height))

    def set_size(self, width: int, height: int) -> None:
        super().set_size(width, height)
        window_frame = self._nswindow.frame()
        rect = self._nswindow.contentRectForFrameRect_(window_frame)
        rect.origin.y += rect.size.height - height
        rect.size.width = width
        rect.size.height = height
        new_frame = self._nswindow.frameRectForContentRect_(rect)
        is_visible = self._nswindow.isVisible()
        self._nswindow.setFrame_display_animate_(new_frame, True, is_visible)
        self.dispatch_event('on_resize', width, height)

    def set_minimum_size(self, width: int, height: int) -> None:
        super().set_minimum_size(width, height)
        if self._nswindow is not None:
            ns_minimum_size = cocoapy.NSSize(*self._minimum_size)
            self._nswindow.setContentMinSize_(ns_minimum_size)

    def set_maximum_size(self, width: int, height: int) -> None:
        super().set_maximum_size(width, height)
        if self._nswindow is not None:
            ns_maximum_size = cocoapy.NSSize(*self._maximum_size)
            self._nswindow.setContentMaxSize_(ns_maximum_size)

    def activate(self):
        if self._nswindow is not None:
            NSApp = NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
            self._nswindow.makeKeyAndOrderFront_(None)

    def set_visible(self, visible: bool=True) -> None:
        super().set_visible(visible)
        if self._nswindow is not None:
            if visible:
                self.dispatch_event('on_resize', self._width, self._height)
                self.dispatch_event('on_show')
                self.dispatch_event('on_expose')
                self._nswindow.makeKeyAndOrderFront_(None)
            else:
                self._nswindow.orderOut_(None)

    def minimize(self):
        self._mouse_in_window = False
        if self._nswindow is not None:
            self._nswindow.miniaturize_(None)

    def maximize(self):
        if self._nswindow is not None:
            self._nswindow.zoom_(None)

    def set_vsync(self, vsync: bool) -> None:
        if pyglet.options['vsync'] is not None:
            vsync = pyglet.options['vsync']
        super().set_vsync(vsync)
        self.context.set_vsync(vsync)

    def _mouse_in_content_rect(self):
        point = NSEvent.mouseLocation()
        window_frame = self._nswindow.frame()
        rect = self._nswindow.contentRectForFrameRect_(window_frame)
        return cocoapy.foundation.NSMouseInRect(point, rect, False)

    def set_mouse_platform_visible(self, platform_visible=None):
        if platform_visible is not None:
            if platform_visible:
                SystemCursor.unhide()
            else:
                SystemCursor.hide()
        elif self._mouse_exclusive:
            SystemCursor.hide()
        elif not self._mouse_in_content_rect():
            NSCursor.arrowCursor().set()
            SystemCursor.unhide()
        elif not self._mouse_visible:
            SystemCursor.hide()
        elif isinstance(self._mouse_cursor, CocoaMouseCursor):
            self._mouse_cursor.set()
            SystemCursor.unhide()
        elif self._mouse_cursor.gl_drawable:
            SystemCursor.hide()
        else:
            NSCursor.arrowCursor().set()
            SystemCursor.unhide()

    def get_system_mouse_cursor(self, name):
        if name == self.CURSOR_DEFAULT:
            return DefaultMouseCursor()
        cursors = {self.CURSOR_CROSSHAIR: 'crosshairCursor', self.CURSOR_HAND: 'pointingHandCursor', self.CURSOR_HELP: 'arrowCursor', self.CURSOR_NO: 'operationNotAllowedCursor', self.CURSOR_SIZE: 'arrowCursor', self.CURSOR_SIZE_UP: 'resizeUpCursor', self.CURSOR_SIZE_UP_RIGHT: 'arrowCursor', self.CURSOR_SIZE_RIGHT: 'resizeRightCursor', self.CURSOR_SIZE_DOWN_RIGHT: 'arrowCursor', self.CURSOR_SIZE_DOWN: 'resizeDownCursor', self.CURSOR_SIZE_DOWN_LEFT: 'arrowCursor', self.CURSOR_SIZE_LEFT: 'resizeLeftCursor', self.CURSOR_SIZE_UP_LEFT: 'arrowCursor', self.CURSOR_SIZE_UP_DOWN: 'resizeUpDownCursor', self.CURSOR_SIZE_LEFT_RIGHT: 'resizeLeftRightCursor', self.CURSOR_TEXT: 'IBeamCursor', self.CURSOR_WAIT: 'arrowCursor', self.CURSOR_WAIT_ARROW: 'arrowCursor'}
        if name not in cursors:
            raise RuntimeError('Unknown cursor name "%s"' % name)
        return CocoaMouseCursor(cursors[name])

    def set_mouse_position(self, x, y, absolute=False):
        if absolute:
            quartz.CGWarpMouseCursorPosition(CGPoint(x, y))
        else:
            screenInfo = self._nswindow.screen().deviceDescription()
            displayID = screenInfo.objectForKey_(cocoapy.get_NSString('NSScreenNumber'))
            displayID = displayID.intValue()
            displayBounds = quartz.CGDisplayBounds(displayID)
            frame = self._nswindow.frame()
            windowOrigin = frame.origin
            x += windowOrigin.x
            y = displayBounds.size.height - windowOrigin.y - y
            quartz.CGDisplayMoveCursorToPoint(displayID, cocoapy.NSPoint(x, y))

    def set_exclusive_mouse(self, exclusive=True):
        super().set_exclusive_mouse(exclusive)
        if exclusive:
            self._mouse_ignore_motion = True
            frame = self._nswindow.frame()
            width, height = (frame.size.width, frame.size.height)
            self.set_mouse_position(width / 2, height / 2)
            quartz.CGAssociateMouseAndMouseCursorPosition(False)
        else:
            quartz.CGAssociateMouseAndMouseCursorPosition(True)
        self.set_mouse_platform_visible()

    def set_exclusive_keyboard(self, exclusive=True):
        super().set_exclusive_keyboard(exclusive)
        if exclusive:
            options = cocoapy.NSApplicationPresentationHideDock | cocoapy.NSApplicationPresentationHideMenuBar | cocoapy.NSApplicationPresentationDisableProcessSwitching | cocoapy.NSApplicationPresentationDisableHideApplication
        else:
            options = cocoapy.NSApplicationPresentationDefault
        NSApp = NSApplication.sharedApplication()
        NSApp.setPresentationOptions_(options)

    def set_clipboard_text(self, text: str):
        with AutoReleasePool():
            pasteboard = NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            array = NSArray.arrayWithObject_(cocoapy.NSPasteboardTypeString)
            pasteboard.declareTypes_owner_(array, None)
            text_nsstring = cocoapy.get_NSString(text)
            pasteboard.setString_forType_(text_nsstring, cocoapy.NSPasteboardTypeString)

    def get_clipboard_text(self) -> str:
        text = ''
        with AutoReleasePool():
            pasteboard = NSPasteboard.generalPasteboard()
            if pasteboard.types().containsObject_(cocoapy.NSPasteboardTypeString):
                text_obj = pasteboard.stringForType_(cocoapy.NSPasteboardTypeString)
                if text_obj:
                    text = text_obj.UTF8String().decode('utf-8')
        return text