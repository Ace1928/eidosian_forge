import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
class Driller(SimpleDialog):

    def __init__(self, parent, manifold):
        self.manifold = manifold
        self.num = 0
        self.max_segments = 6
        self.result = []
        style = SnapPyStyle()
        self.root = root = Tk_.Toplevel(parent, class_='SnapPy', bg=style.windowBG)
        title = 'Drill'
        root.title(title)
        root.iconname(title)
        root.bind('<Return>', self.handle_return)
        top_frame = ttk.Frame(self.root)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_rowconfigure(2, weight=1)
        msg_font = Font(family=style.font_info['family'], weight='bold', size=int(style.font_info['size'] * 1.2))
        msg = ttk.Label(top_frame, font=msg_font, text='Choose which curves to drill out:')
        msg.grid(row=0, column=0, pady=10)
        segment_frame = ttk.Frame(top_frame)
        self.segment_var = segment_var = Tk_.StringVar(root)
        segment_var.set(str(self.max_segments))
        ttk.Label(segment_frame, text='Max segments: ').pack(side=Tk_.LEFT, padx=4)
        self.segment_entry = segment_entry = ttk.Entry(segment_frame, takefocus=False, width=2, textvariable=segment_var, validate='focusout', validatecommand=(root.register(self.validate_segments), '%P'))
        segment_entry.pack(side=Tk_.LEFT)
        segment_frame.grid(row=1, column=0, pady=2)
        self.curves = curves = ttk.Treeview(top_frame, selectmode='extended', columns=['index', 'parity', 'length'], show='headings')
        curves.heading('index', text='#')
        curves.column('index', stretch=False, width=20)
        curves.heading('parity', text='Parity')
        curves.column('parity', stretch=False, width=80)
        curves.heading('length', text='Length')
        curves.column('length', stretch=True, width=460)
        curves.bind('<Double-Button-1>', self.drill)
        self.curves.grid(row=2, column=0, padx=6, pady=6, sticky=Tk_.NSEW)
        self.show_curves()
        top_frame.pack(fill=Tk_.BOTH, expand=1)
        button_frame = ttk.Frame(self.root)
        button = ttk.Button(button_frame, text='Drill', command=self.drill, default='active')
        button.pack(side=Tk_.LEFT, padx=6)
        button = ttk.Button(button_frame, text='Cancel', command=self.cancel)
        button.pack(side=Tk_.LEFT, padx=6)
        button_frame.pack(pady=6)
        self.root.protocol('WM_DELETE_WINDOW', self.wm_delete_window)
        _place_window(self.root, parent)

    def show_curves(self):
        self.curves.delete(*self.curves.get_children())
        for curve in self.manifold.dual_curves(max_segments=self.max_segments):
            n = curve['index']
            parity = '+' if curve['parity'] == 1 else '-'
            length = Number(curve['filled_length'], precision=25)
            self.curves.insert('', 'end', values=(n, parity, length))

    def handle_return(self, event):
        if event.widget != self.segment_entry:
            self.drill()
        else:
            self.curves.focus_set()

    def drill(self, event=None):
        self.result = [self.curves.index(x) for x in self.curves.selection()]
        self.root.quit()

    def cancel(self):
        self.root.quit()

    def validate_segments(self, P):
        try:
            new_max = int(P)
            if self.max_segments != new_max:
                self.max_segments = new_max
                self.segment_var.set(str(self.max_segments))
                self.show_curves()
        except ValueError:
            self.root.after_idle(self.segment_var.set, str(self.max_segments))
            return False
        return True