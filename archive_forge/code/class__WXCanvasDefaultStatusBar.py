from pidWxDc import PiddleWxDc
from wxPython.wx import *
class _WXCanvasDefaultStatusBar(wxStatusBar):
    """This status bar displays clear and quit buttons, as well as the
   location of the mouse and if the left button is down."""

    def __init__(self, canvas, pos, size):
        wxStatusBar.__init__(self, canvas.window, -1, pos, size)
        self.parentwindow = canvas.window
        self.parentwindow.SetStatusBar(self)
        self.SetFieldsCount(3)
        self.sizeChanged = false
        self.text = ''
        self.old_text = ''
        self.canvas = canvas
        qID = NewId()
        cID = NewId()
        sID = NewId()
        bID = NewId()
        self.quitButton = wxButton(self, qID, 'Quit')
        self.clearButton = wxButton(self, cID, 'Clear')
        self.click = wxCheckBox(self, bID, 'Click')
        self.Reposition()
        EVT_BUTTON(self, qID, canvas._OnQuit)
        EVT_BUTTON(self, cID, canvas._OnClear)
        EVT_PAINT(self, self.repaint)
        EVT_SIZE(self, self.OnSize)
        EVT_IDLE(self, self.OnIdle)

    def repaint(self, event):
        dc = wxPaintDC(self)
        self.draw(dc)

    def redraw(self):
        dc = wxClientDC(self)
        self.draw(dc)

    def draw(self, dc):
        field = self.GetFieldRect(1)
        extents = dc.GetTextExtent(self.old_text)
        dc.SetPen(wxTRANSPARENT_PEN)
        dc.SetBrush(dc.GetBackground())
        field = self.GetFieldRect(2)
        dc.DrawRectangle(field.x, field.y, field.width, field.height)
        dc.SetFont(wxFont(9, wxDEFAULT, wxNORMAL, wxNORMAL))
        extents = dc.GetTextExtent(self.text)
        dc.DrawText(self.text, field.x + field.width / 2 - extents[0] / 2, field.y)
        self.old_text = self.text

    def SetStatusText(self, s):
        self.text = s
        self.redraw()

    def OnOver(self, x, y):
        self.text = repr(x) + ',' + repr(y)
        self.redraw()

    def OnClick(self, x, y):
        self.text = repr(x) + ',' + repr(y)
        self.click.SetValue(true)
        self.redraw()

    def OnLeaveWindow(self):
        self.click.SetValue(false)
        self.text = ''
        self.redraw()

    def OnClickUp(self, x, y):
        self.click.SetValue(false)

    def OnSize(self, evt):
        self.Reposition()
        self.sizeChanged = true

    def OnIdle(self, evt):
        if self.sizeChanged:
            self.Reposition()

    def Reposition(self):
        field = self.GetFieldRect(0)
        self.quitButton.SetPosition(wxPoint(field.x, field.y))
        self.quitButton.SetSize(wxSize(field.width / 2, field.height))
        self.clearButton.SetPosition(wxPoint(field.x + field.width / 2, field.y))
        self.clearButton.SetSize(wxSize(field.width / 2, field.height))
        field = self.GetFieldRect(1)
        self.click.SetPosition(wxPoint(field.x + field.width / 2 - 20, field.y))
        self.click.SetSize(wxSize(100, field.height))
        self.sizeChanged = false