from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class CanvasFrame:
    """
    A ``Tkinter`` frame containing a canvas and scrollbars.
    ``CanvasFrame`` uses a ``ScrollWatcherWidget`` to ensure that all of
    the canvas widgets contained on its canvas are within its
    scrollregion.  In order for ``CanvasFrame`` to make these checks,
    all canvas widgets must be registered with ``add_widget`` when they
    are added to the canvas; and destroyed with ``destroy_widget`` when
    they are no longer needed.

    If a ``CanvasFrame`` is created with no parent, then it will create
    its own main window, including a "Done" button and a "Print"
    button.
    """

    def __init__(self, parent=None, **kw):
        """
        Create a new ``CanvasFrame``.

        :type parent: Tkinter.BaseWidget or Tkinter.Tk
        :param parent: The parent ``Tkinter`` widget.  If no parent is
            specified, then ``CanvasFrame`` will create a new main
            window.
        :param kw: Keyword arguments for the new ``Canvas``.  See the
            documentation for ``Tkinter.Canvas`` for more information.
        """
        if parent is None:
            self._parent = Tk()
            self._parent.title('NLTK')
            self._parent.bind('<Control-p>', lambda e: self.print_to_file())
            self._parent.bind('<Control-x>', self.destroy)
            self._parent.bind('<Control-q>', self.destroy)
        else:
            self._parent = parent
        self._frame = frame = Frame(self._parent)
        self._canvas = canvas = Canvas(frame, **kw)
        xscrollbar = Scrollbar(self._frame, orient='horizontal')
        yscrollbar = Scrollbar(self._frame, orient='vertical')
        xscrollbar['command'] = canvas.xview
        yscrollbar['command'] = canvas.yview
        canvas['xscrollcommand'] = xscrollbar.set
        canvas['yscrollcommand'] = yscrollbar.set
        yscrollbar.pack(fill='y', side='right')
        xscrollbar.pack(fill='x', side='bottom')
        canvas.pack(expand=1, fill='both', side='left')
        scrollregion = '0 0 {} {}'.format(canvas['width'], canvas['height'])
        canvas['scrollregion'] = scrollregion
        self._scrollwatcher = ScrollWatcherWidget(canvas)
        if parent is None:
            self.pack(expand=1, fill='both')
            self._init_menubar()

    def _init_menubar(self):
        menubar = Menu(self._parent)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Print to Postscript', underline=0, command=self.print_to_file, accelerator='Ctrl-p')
        filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
        menubar.add_cascade(label='File', underline=0, menu=filemenu)
        self._parent.config(menu=menubar)

    def print_to_file(self, filename=None):
        """
        Print the contents of this ``CanvasFrame`` to a postscript
        file.  If no filename is given, then prompt the user for one.

        :param filename: The name of the file to print the tree to.
        :type filename: str
        :rtype: None
        """
        if filename is None:
            ftypes = [('Postscript files', '.ps'), ('All files', '*')]
            filename = asksaveasfilename(filetypes=ftypes, defaultextension='.ps')
            if not filename:
                return
        x0, y0, w, h = self.scrollregion()
        postscript = self._canvas.postscript(x=x0, y=y0, width=w + 2, height=h + 2, pagewidth=w + 2, pageheight=h + 2, pagex=0, pagey=0)
        postscript = postscript.replace(' 0 scalefont ', ' 9 scalefont ')
        with open(filename, 'wb') as f:
            f.write(postscript.encode('utf8'))

    def scrollregion(self):
        """
        :return: The current scroll region for the canvas managed by
            this ``CanvasFrame``.
        :rtype: 4-tuple of int
        """
        x1, y1, x2, y2 = self._canvas['scrollregion'].split()
        return (int(x1), int(y1), int(x2), int(y2))

    def canvas(self):
        """
        :return: The canvas managed by this ``CanvasFrame``.
        :rtype: Tkinter.Canvas
        """
        return self._canvas

    def add_widget(self, canvaswidget, x=None, y=None):
        """
        Register a canvas widget with this ``CanvasFrame``.  The
        ``CanvasFrame`` will ensure that this canvas widget is always
        within the ``Canvas``'s scrollregion.  If no coordinates are
        given for the canvas widget, then the ``CanvasFrame`` will
        attempt to find a clear area of the canvas for it.

        :type canvaswidget: CanvasWidget
        :param canvaswidget: The new canvas widget.  ``canvaswidget``
            must have been created on this ``CanvasFrame``'s canvas.
        :type x: int
        :param x: The initial x coordinate for the upper left hand
            corner of ``canvaswidget``, in the canvas's coordinate
            space.
        :type y: int
        :param y: The initial y coordinate for the upper left hand
            corner of ``canvaswidget``, in the canvas's coordinate
            space.
        """
        if x is None or y is None:
            x, y = self._find_room(canvaswidget, x, y)
        x1, y1, x2, y2 = canvaswidget.bbox()
        canvaswidget.move(x - x1, y - y1)
        self._scrollwatcher.add_child(canvaswidget)

    def _find_room(self, widget, desired_x, desired_y):
        """
        Try to find a space for a given widget.
        """
        left, top, right, bot = self.scrollregion()
        w = widget.width()
        h = widget.height()
        if w >= right - left:
            return (0, 0)
        if h >= bot - top:
            return (0, 0)
        x1, y1, x2, y2 = widget.bbox()
        widget.move(left - x2 - 50, top - y2 - 50)
        if desired_x is not None:
            x = desired_x
            for y in range(top, bot - h, int((bot - top - h) / 10)):
                if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                    return (x, y)
        if desired_y is not None:
            y = desired_y
            for x in range(left, right - w, int((right - left - w) / 10)):
                if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                    return (x, y)
        for y in range(top, bot - h, int((bot - top - h) / 10)):
            for x in range(left, right - w, int((right - left - w) / 10)):
                if not self._canvas.find_overlapping(x - 5, y - 5, x + w + 5, y + h + 5):
                    return (x, y)
        return (0, 0)

    def destroy_widget(self, canvaswidget):
        """
        Remove a canvas widget from this ``CanvasFrame``.  This
        deregisters the canvas widget, and destroys it.
        """
        self.remove_widget(canvaswidget)
        canvaswidget.destroy()

    def remove_widget(self, canvaswidget):
        self._scrollwatcher.remove_child(canvaswidget)

    def pack(self, cnf={}, **kw):
        """
        Pack this ``CanvasFrame``.  See the documentation for
        ``Tkinter.Pack`` for more information.
        """
        self._frame.pack(cnf, **kw)

    def destroy(self, *e):
        """
        Destroy this ``CanvasFrame``.  If this ``CanvasFrame`` created a
        top-level window, then this will close that window.
        """
        if self._parent is None:
            return
        self._parent.destroy()
        self._parent = None

    def mainloop(self, *args, **kwargs):
        """
        Enter the Tkinter mainloop.  This function must be called if
        this frame is created from a non-interactive program (e.g.
        from a secript); otherwise, the frame will close as soon as
        the script completes.
        """
        if in_idle():
            return
        self._parent.mainloop(*args, **kwargs)