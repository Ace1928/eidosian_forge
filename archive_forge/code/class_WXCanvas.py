from pidWxDc import PiddleWxDc
from wxPython.wx import *
class WXCanvas(PiddleWxDc):

    def __init__(self, size=(300, 300), name='piddleWX', status_bar=None, interactive=1, show_status=1):
        """Works like all other PIDDLE canvases, except with extra interactive controls.
       interactive is set if the canvas is to use the interactive parts of
       the PIDDLE API, and show_status controls if the default status bar is
       shown"""
        window = wxFrame(NULL, -1, 'WXCanvas', wxPyDefaultPosition, wxSize(size[0], size[1]))
        window.Show(true)
        self.window = window
        CSize = window.GetClientSizeTuple()
        if show_status:
            status_area = 20
        else:
            status_area = 0
        window.SetSize(wxSize(size[0] + (size[0] - CSize[0]), size[1] + (size[1] - CSize[1] + status_area)))
        bmp = wxEmptyBitmap(size[0], size[1])
        MemDc = wxMemoryDC()
        MemDc.SelectObject(bmp)
        MemDc.Clear()
        self.MemDc = MemDc
        PiddleWxDc.__init__(self, MemDc, size, name)
        self.window = window
        self.size = size
        if status_bar is None:
            self.sb = _WXCanvasDefaultStatusBar(self, wxPoint(0, size[1]), wxSize(size[0], 20))
        else:
            self.sb = status_bar
        if show_status == 0:
            self.sb.Show(false)
        self.sb.redraw()

        def ignoreClick(canvas, x, y):
            canvas.sb.OnClick(x, y)
        self.onClick = ignoreClick

        def ignoreOver(canvas, x, y):
            canvas.sb.OnOver(x, y)
        self.onOver = ignoreOver

        def ignoreKey(canvas, key, modifiers):
            pass
        self.onKey = ignoreKey

        def ignoreClickUp(canvas, x, y):
            canvas.sb.OnClickUp(x, y)
        self.onClickUp = ignoreClickUp
        self.interactive = interactive
        EVT_PAINT(window, self._OnPaint)
        EVT_LEFT_DOWN(window, self._OnClick)
        EVT_LEFT_UP(window, self._OnClickUp)
        EVT_MOTION(window, self._OnOver)
        EVT_CHAR(window, self._OnKey)
        EVT_LEAVE_WINDOW(window, self._OnLeaveWindow)

        def leaveWindow(canvas):
            canvas.sb.OnLeaveWindow()
        self.onLeaveWindow = leaveWindow

    def _OnClick(self, event):
        if self.interactive == false:
            return None
        if event.GetY() <= self.size[1]:
            self.onClick(self, event.GetX(), event.GetY())

    def _OnClickUp(self, event):
        if self.interactive == false:
            return None
        self.onClickUp(self, event.GetX(), event.GetY())

    def _OnOver(self, event):
        if self.interactive == false:
            return None
        if event.GetY() <= self.size[1]:
            self.onOver(self, event.GetX(), event.GetY())

    def _OnLeaveWindow(self, event):
        if self.interactive == false:
            return None
        self.onLeaveWindow(self)

    def _OnKey(self, event):
        code = event.KeyCode()
        key = None
        if code >= 0 and code < 256:
            key = chr(event.KeyCode())
        modifier = []
        if event.ControlDown():
            modifier.append('modControl')
        if event.ShiftDown():
            modifier.append('modshift')
        self.onKey(self, key, tuple(modifier))

    def _OnPaint(self, event):
        dc = wxPaintDC(self.window)
        dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)
        del dc

    def _OnQuit(self, event):
        """Closes the canvas.  Call to return control your application"""
        self.window.Close()

    def _OnClear(self, event):
        """Clears the canvas by emptying the memory buffer, and redrawing"""
        self.MemDc.Clear()
        dc = wxClientDC(self.window)
        dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)

    def isInteractive(self):
        """Returns 1 if onClick and onOver events are possible, 0 otherwise."""
        return self.interactive

    def canUpdate(self):
        """Returns 1 if the drawing can be meaningfully updated over time (e.g.,
       screen graphics), 0 otherwise (e.g., drawing to a file)."""
        return 1

    def clear(self):
        self.Clear()
        dc = wxClientDC(self.window)
        dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)

    def flush(self):
        """Copies the contents of the memory buffer to the screen and enters the
       application main loop"""
        dc = wxClientDC(self.window)
        dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)
        del dc

    def setInfoLine(self, s):
        """For interactive Canvases, displays the given string in the 'info
       line' somewhere where the user can probably see it."""
        if self.sb is not None:
            self.sb.SetStatusText(s)