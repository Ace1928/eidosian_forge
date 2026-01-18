import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
class ChartComparer:
    """

    :ivar _root: The root window

    :ivar _charts: A dictionary mapping names to charts.  When
        charts are loaded, they are added to this dictionary.

    :ivar _left_chart: The left ``Chart``.
    :ivar _left_name: The name ``_left_chart`` (derived from filename)
    :ivar _left_matrix: The ``ChartMatrixView`` for ``_left_chart``
    :ivar _left_selector: The drop-down ``MutableOptionsMenu`` used
          to select ``_left_chart``.

    :ivar _right_chart: The right ``Chart``.
    :ivar _right_name: The name ``_right_chart`` (derived from filename)
    :ivar _right_matrix: The ``ChartMatrixView`` for ``_right_chart``
    :ivar _right_selector: The drop-down ``MutableOptionsMenu`` used
          to select ``_right_chart``.

    :ivar _out_chart: The out ``Chart``.
    :ivar _out_name: The name ``_out_chart`` (derived from filename)
    :ivar _out_matrix: The ``ChartMatrixView`` for ``_out_chart``
    :ivar _out_label: The label for ``_out_chart``.

    :ivar _op_label: A Label containing the most recent operation.
    """
    _OPSYMBOL = {'-': '-', 'and': SymbolWidget.SYMBOLS['intersection'], 'or': SymbolWidget.SYMBOLS['union']}

    def __init__(self, *chart_filenames):
        faketok = [''] * 8
        self._emptychart = Chart(faketok)
        self._left_name = 'None'
        self._right_name = 'None'
        self._left_chart = self._emptychart
        self._right_chart = self._emptychart
        self._charts = {'None': self._emptychart}
        self._out_chart = self._emptychart
        self._operator = None
        self._root = Tk()
        self._root.title('Chart Comparison')
        self._root.bind('<Control-q>', self.destroy)
        self._root.bind('<Control-x>', self.destroy)
        self._init_menubar(self._root)
        self._init_chartviews(self._root)
        self._init_divider(self._root)
        self._init_buttons(self._root)
        self._init_bindings(self._root)
        for filename in chart_filenames:
            self.load_chart(filename)

    def destroy(self, *e):
        if self._root is None:
            return
        try:
            self._root.destroy()
        except:
            pass
        self._root = None

    def mainloop(self, *args, **kwargs):
        return
        self._root.mainloop(*args, **kwargs)

    def _init_menubar(self, root):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load Chart', accelerator='Ctrl-o', underline=0, command=self.load_chart_dialog)
        filemenu.add_command(label='Save Output', accelerator='Ctrl-s', underline=0, command=self.save_chart_dialog)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
        menubar.add_cascade(label='File', underline=0, menu=filemenu)
        opmenu = Menu(menubar, tearoff=0)
        opmenu.add_command(label='Intersection', command=self._intersection, accelerator='+')
        opmenu.add_command(label='Union', command=self._union, accelerator='*')
        opmenu.add_command(label='Difference', command=self._difference, accelerator='-')
        opmenu.add_separator()
        opmenu.add_command(label='Swap Charts', command=self._swapcharts)
        menubar.add_cascade(label='Compare', underline=0, menu=opmenu)
        self._root.config(menu=menubar)

    def _init_divider(self, root):
        divider = Frame(root, border=2, relief='sunken')
        divider.pack(side='top', fill='x', ipady=2)

    def _init_chartviews(self, root):
        opfont = ('symbol', -36)
        eqfont = ('helvetica', -36)
        frame = Frame(root, background='#c0c0c0')
        frame.pack(side='top', expand=1, fill='both')
        cv1_frame = Frame(frame, border=3, relief='groove')
        cv1_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._left_selector = MutableOptionMenu(cv1_frame, list(self._charts.keys()), command=self._select_left)
        self._left_selector.pack(side='top', pady=5, fill='x')
        self._left_matrix = ChartMatrixView(cv1_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._left_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._left_matrix.add_callback('select', self.select_edge)
        self._left_matrix.add_callback('select_cell', self.select_cell)
        self._left_matrix.inactivate()
        self._op_label = Label(frame, text=' ', width=3, background='#c0c0c0', font=opfont)
        self._op_label.pack(side='left', padx=5, pady=5)
        cv2_frame = Frame(frame, border=3, relief='groove')
        cv2_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._right_selector = MutableOptionMenu(cv2_frame, list(self._charts.keys()), command=self._select_right)
        self._right_selector.pack(side='top', pady=5, fill='x')
        self._right_matrix = ChartMatrixView(cv2_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._right_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._right_matrix.add_callback('select', self.select_edge)
        self._right_matrix.add_callback('select_cell', self.select_cell)
        self._right_matrix.inactivate()
        Label(frame, text='=', width=3, background='#c0c0c0', font=eqfont).pack(side='left', padx=5, pady=5)
        out_frame = Frame(frame, border=3, relief='groove')
        out_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
        self._out_label = Label(out_frame, text='Output')
        self._out_label.pack(side='top', pady=9)
        self._out_matrix = ChartMatrixView(out_frame, self._emptychart, toplevel=False, show_numedges=True)
        self._out_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
        self._out_matrix.add_callback('select', self.select_edge)
        self._out_matrix.add_callback('select_cell', self.select_cell)
        self._out_matrix.inactivate()

    def _init_buttons(self, root):
        buttons = Frame(root)
        buttons.pack(side='bottom', pady=5, fill='x', expand=0)
        Button(buttons, text='Intersection', command=self._intersection).pack(side='left')
        Button(buttons, text='Union', command=self._union).pack(side='left')
        Button(buttons, text='Difference', command=self._difference).pack(side='left')
        Frame(buttons, width=20).pack(side='left')
        Button(buttons, text='Swap Charts', command=self._swapcharts).pack(side='left')
        Button(buttons, text='Detach Output', command=self._detach_out).pack(side='right')

    def _init_bindings(self, root):
        root.bind('<Control-o>', self.load_chart_dialog)

    def _select_left(self, name):
        self._left_name = name
        self._left_chart = self._charts[name]
        self._left_matrix.set_chart(self._left_chart)
        if name == 'None':
            self._left_matrix.inactivate()
        self._apply_op()

    def _select_right(self, name):
        self._right_name = name
        self._right_chart = self._charts[name]
        self._right_matrix.set_chart(self._right_chart)
        if name == 'None':
            self._right_matrix.inactivate()
        self._apply_op()

    def _apply_op(self):
        if self._operator == '-':
            self._difference()
        elif self._operator == 'or':
            self._union()
        elif self._operator == 'and':
            self._intersection()
    CHART_FILE_TYPES = [('Pickle file', '.pickle'), ('All files', '*')]

    def save_chart_dialog(self, *args):
        filename = asksaveasfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            with open(filename, 'wb') as outfile:
                pickle.dump(self._out_chart, outfile)
        except Exception as e:
            showerror('Error Saving Chart', f'Unable to open file: {filename!r}\n{e}')

    def load_chart_dialog(self, *args):
        filename = askopenfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
        if not filename:
            return
        try:
            self.load_chart(filename)
        except Exception as e:
            showerror('Error Loading Chart', f'Unable to open file: {filename!r}\n{e}')

    def load_chart(self, filename):
        with open(filename, 'rb') as infile:
            chart = pickle.load(infile)
        name = os.path.basename(filename)
        if name.endswith('.pickle'):
            name = name[:-7]
        if name.endswith('.chart'):
            name = name[:-6]
        self._charts[name] = chart
        self._left_selector.add(name)
        self._right_selector.add(name)
        if self._left_chart is self._emptychart:
            self._left_selector.set(name)
        elif self._right_chart is self._emptychart:
            self._right_selector.set(name)

    def _update_chartviews(self):
        self._left_matrix.update()
        self._right_matrix.update()
        self._out_matrix.update()

    def select_edge(self, edge):
        if edge in self._left_chart:
            self._left_matrix.markonly_edge(edge)
        else:
            self._left_matrix.unmark_edge()
        if edge in self._right_chart:
            self._right_matrix.markonly_edge(edge)
        else:
            self._right_matrix.unmark_edge()
        if edge in self._out_chart:
            self._out_matrix.markonly_edge(edge)
        else:
            self._out_matrix.unmark_edge()

    def select_cell(self, i, j):
        self._left_matrix.select_cell(i, j)
        self._right_matrix.select_cell(i, j)
        self._out_matrix.select_cell(i, j)

    def _difference(self):
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            if edge not in self._right_chart:
                out_chart.insert(edge, [])
        self._update('-', out_chart)

    def _intersection(self):
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            if edge in self._right_chart:
                out_chart.insert(edge, [])
        self._update('and', out_chart)

    def _union(self):
        if not self._checkcompat():
            return
        out_chart = Chart(self._left_chart.tokens())
        for edge in self._left_chart:
            out_chart.insert(edge, [])
        for edge in self._right_chart:
            out_chart.insert(edge, [])
        self._update('or', out_chart)

    def _swapcharts(self):
        left, right = (self._left_name, self._right_name)
        self._left_selector.set(right)
        self._right_selector.set(left)

    def _checkcompat(self):
        if self._left_chart.tokens() != self._right_chart.tokens() or self._left_chart.property_names() != self._right_chart.property_names() or self._left_chart == self._emptychart or (self._right_chart == self._emptychart):
            self._out_chart = self._emptychart
            self._out_matrix.set_chart(self._out_chart)
            self._out_matrix.inactivate()
            self._out_label['text'] = 'Output'
            return False
        else:
            return True

    def _update(self, operator, out_chart):
        self._operator = operator
        self._op_label['text'] = self._OPSYMBOL[operator]
        self._out_chart = out_chart
        self._out_matrix.set_chart(out_chart)
        self._out_label['text'] = '{} {} {}'.format(self._left_name, self._operator, self._right_name)

    def _clear_out_chart(self):
        self._out_chart = self._emptychart
        self._out_matrix.set_chart(self._out_chart)
        self._op_label['text'] = ' '
        self._out_matrix.inactivate()

    def _detach_out(self):
        ChartMatrixView(self._root, self._out_chart, title=self._out_label['text'])