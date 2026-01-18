import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
class MultiListbox(Frame):
    """
    A multi-column listbox, where the current selection applies to an
    entire row.  Based on the MultiListbox Tkinter widget
    recipe from the Python Cookbook (https://code.activestate.com/recipes/52266/)

    For the most part, ``MultiListbox`` methods delegate to its
    contained listboxes.  For any methods that do not have docstrings,
    see ``Tkinter.Listbox`` for a description of what that method does.
    """
    FRAME_CONFIG = dict(background='#888', takefocus=True, highlightthickness=1)
    LABEL_CONFIG = dict(borderwidth=1, relief='raised', font='helvetica -16 bold', background='#444', foreground='white')
    LISTBOX_CONFIG = dict(borderwidth=1, selectborderwidth=0, highlightthickness=0, exportselection=False, selectbackground='#888', activestyle='none', takefocus=False)

    def __init__(self, master, columns, column_weights=None, cnf={}, **kw):
        """
        Construct a new multi-column listbox widget.

        :param master: The widget that should contain the new
            multi-column listbox.

        :param columns: Specifies what columns should be included in
            the new multi-column listbox.  If ``columns`` is an integer,
            then it is the number of columns to include.  If it is
            a list, then its length indicates the number of columns
            to include; and each element of the list will be used as
            a label for the corresponding column.

        :param cnf, kw: Configuration parameters for this widget.
            Use ``label_*`` to configure all labels; and ``listbox_*``
            to configure all listboxes.  E.g.:
                >>> root = Tk()  # doctest: +SKIP
                >>> MultiListbox(root, ["Subject", "Sender", "Date"], label_foreground='red').pack()  # doctest: +SKIP
        """
        if isinstance(columns, int):
            columns = list(range(columns))
            include_labels = False
        else:
            include_labels = True
        if len(columns) == 0:
            raise ValueError('Expected at least one column')
        self._column_names = tuple(columns)
        self._listboxes = []
        self._labels = []
        if column_weights is None:
            column_weights = [1] * len(columns)
        elif len(column_weights) != len(columns):
            raise ValueError('Expected one column_weight for each column')
        self._column_weights = column_weights
        Frame.__init__(self, master, **self.FRAME_CONFIG)
        self.grid_rowconfigure(1, weight=1)
        for i, label in enumerate(self._column_names):
            self.grid_columnconfigure(i, weight=column_weights[i])
            if include_labels:
                l = Label(self, text=label, **self.LABEL_CONFIG)
                self._labels.append(l)
                l.grid(column=i, row=0, sticky='news', padx=0, pady=0)
                l.column_index = i
            lb = Listbox(self, **self.LISTBOX_CONFIG)
            self._listboxes.append(lb)
            lb.grid(column=i, row=1, sticky='news', padx=0, pady=0)
            lb.column_index = i
            lb.bind('<Button-1>', self._select)
            lb.bind('<B1-Motion>', self._select)
            lb.bind('<Button-4>', lambda e: self._scroll(-1))
            lb.bind('<Button-5>', lambda e: self._scroll(+1))
            lb.bind('<MouseWheel>', lambda e: self._scroll(e.delta))
            lb.bind('<Button-2>', lambda e: self.scan_mark(e.x, e.y))
            lb.bind('<B2-Motion>', lambda e: self.scan_dragto(e.x, e.y))
            lb.bind('<B1-Leave>', lambda e: 'break')
            lb.bind('<Button-1>', self._resize_column)
        self.bind('<Button-1>', self._resize_column)
        self.bind('<Up>', lambda e: self.select(delta=-1))
        self.bind('<Down>', lambda e: self.select(delta=1))
        self.bind('<Prior>', lambda e: self.select(delta=-self._pagesize()))
        self.bind('<Next>', lambda e: self.select(delta=self._pagesize()))
        self.configure(cnf, **kw)

    def _resize_column(self, event):
        """
        Callback used to resize a column of the table.  Return ``True``
        if the column is actually getting resized (if the user clicked
        on the far left or far right 5 pixels of a label); and
        ``False`` otherwies.
        """
        if event.widget.bind('<ButtonRelease>'):
            return False
        self._resize_column_index = None
        if event.widget is self:
            for i, lb in enumerate(self._listboxes):
                if abs(event.x - (lb.winfo_x() + lb.winfo_width())) < 10:
                    self._resize_column_index = i
        elif event.x > event.widget.winfo_width() - 5:
            self._resize_column_index = event.widget.column_index
        elif event.x < 5 and event.widget.column_index != 0:
            self._resize_column_index = event.widget.column_index - 1
        if self._resize_column_index is not None:
            event.widget.bind('<Motion>', self._resize_column_motion_cb)
            event.widget.bind('<ButtonRelease-%d>' % event.num, self._resize_column_buttonrelease_cb)
            return True
        else:
            return False

    def _resize_column_motion_cb(self, event):
        lb = self._listboxes[self._resize_column_index]
        charwidth = lb.winfo_width() / lb['width']
        x1 = event.x + event.widget.winfo_x()
        x2 = lb.winfo_x() + lb.winfo_width()
        lb['width'] = max(3, lb['width'] + (x1 - x2) // charwidth)

    def _resize_column_buttonrelease_cb(self, event):
        event.widget.unbind('<ButtonRelease-%d>' % event.num)
        event.widget.unbind('<Motion>')

    @property
    def column_names(self):
        """
        A tuple containing the names of the columns used by this
        multi-column listbox.
        """
        return self._column_names

    @property
    def column_labels(self):
        """
        A tuple containing the ``Tkinter.Label`` widgets used to
        display the label of each column.  If this multi-column
        listbox was created without labels, then this will be an empty
        tuple.  These widgets will all be augmented with a
        ``column_index`` attribute, which can be used to determine
        which column they correspond to.  This can be convenient,
        e.g., when defining callbacks for bound events.
        """
        return tuple(self._labels)

    @property
    def listboxes(self):
        """
        A tuple containing the ``Tkinter.Listbox`` widgets used to
        display individual columns.  These widgets will all be
        augmented with a ``column_index`` attribute, which can be used
        to determine which column they correspond to.  This can be
        convenient, e.g., when defining callbacks for bound events.
        """
        return tuple(self._listboxes)

    def _select(self, e):
        i = e.widget.nearest(e.y)
        self.selection_clear(0, 'end')
        self.selection_set(i)
        self.activate(i)
        self.focus()

    def _scroll(self, delta):
        for lb in self._listboxes:
            lb.yview_scroll(delta, 'unit')
        return 'break'

    def _pagesize(self):
        """:return: The number of rows that makes up one page"""
        return int(self.index('@0,1000000')) - int(self.index('@0,0'))

    def select(self, index=None, delta=None, see=True):
        """
        Set the selected row.  If ``index`` is specified, then select
        row ``index``.  Otherwise, if ``delta`` is specified, then move
        the current selection by ``delta`` (negative numbers for up,
        positive numbers for down).  This will not move the selection
        past the top or the bottom of the list.

        :param see: If true, then call ``self.see()`` with the newly
            selected index, to ensure that it is visible.
        """
        if index is not None and delta is not None:
            raise ValueError('specify index or delta, but not both')
        if delta is not None:
            if len(self.curselection()) == 0:
                index = -1 + delta
            else:
                index = int(self.curselection()[0]) + delta
        self.selection_clear(0, 'end')
        if index is not None:
            index = min(max(index, 0), self.size() - 1)
            self.selection_set(index)
            if see:
                self.see(index)

    def configure(self, cnf={}, **kw):
        """
        Configure this widget.  Use ``label_*`` to configure all
        labels; and ``listbox_*`` to configure all listboxes.  E.g.:

                >>> master = Tk()  # doctest: +SKIP
                >>> mlb = MultiListbox(master, 5)  # doctest: +SKIP
                >>> mlb.configure(label_foreground='red')  # doctest: +SKIP
                >>> mlb.configure(listbox_foreground='red')  # doctest: +SKIP
        """
        cnf = dict(list(cnf.items()) + list(kw.items()))
        for key, val in list(cnf.items()):
            if key.startswith('label_') or key.startswith('label-'):
                for label in self._labels:
                    label.configure({key[6:]: val})
            elif key.startswith('listbox_') or key.startswith('listbox-'):
                for listbox in self._listboxes:
                    listbox.configure({key[8:]: val})
            else:
                Frame.configure(self, {key: val})

    def __setitem__(self, key, val):
        """
        Configure this widget.  This is equivalent to
        ``self.configure({key,val``)}.  See ``configure()``.
        """
        self.configure({key: val})

    def rowconfigure(self, row_index, cnf={}, **kw):
        """
        Configure all table cells in the given row.  Valid keyword
        arguments are: ``background``, ``bg``, ``foreground``, ``fg``,
        ``selectbackground``, ``selectforeground``.
        """
        for lb in self._listboxes:
            lb.itemconfigure(row_index, cnf, **kw)

    def columnconfigure(self, col_index, cnf={}, **kw):
        """
        Configure all table cells in the given column.  Valid keyword
        arguments are: ``background``, ``bg``, ``foreground``, ``fg``,
        ``selectbackground``, ``selectforeground``.
        """
        lb = self._listboxes[col_index]
        cnf = dict(list(cnf.items()) + list(kw.items()))
        for key, val in list(cnf.items()):
            if key in ('background', 'bg', 'foreground', 'fg', 'selectbackground', 'selectforeground'):
                for i in range(lb.size()):
                    lb.itemconfigure(i, {key: val})
            else:
                lb.configure({key: val})

    def itemconfigure(self, row_index, col_index, cnf=None, **kw):
        """
        Configure the table cell at the given row and column.  Valid
        keyword arguments are: ``background``, ``bg``, ``foreground``,
        ``fg``, ``selectbackground``, ``selectforeground``.
        """
        lb = self._listboxes[col_index]
        return lb.itemconfigure(row_index, cnf, **kw)

    def insert(self, index, *rows):
        """
        Insert the given row or rows into the table, at the given
        index.  Each row value should be a tuple of cell values, one
        for each column in the row.  Index may be an integer or any of
        the special strings (such as ``'end'``) accepted by
        ``Tkinter.Listbox``.
        """
        for elt in rows:
            if len(elt) != len(self._column_names):
                raise ValueError('rows should be tuples whose length is equal to the number of columns')
        for lb, elts in zip(self._listboxes, list(zip(*rows))):
            lb.insert(index, *elts)

    def get(self, first, last=None):
        """
        Return the value(s) of the specified row(s).  If ``last`` is
        not specified, then return a single row value; otherwise,
        return a list of row values.  Each row value is a tuple of
        cell values, one for each column in the row.
        """
        values = [lb.get(first, last) for lb in self._listboxes]
        if last:
            return [tuple(row) for row in zip(*values)]
        else:
            return tuple(values)

    def bbox(self, row, col):
        """
        Return the bounding box for the given table cell, relative to
        this widget's top-left corner.  The bounding box is a tuple
        of integers ``(left, top, width, height)``.
        """
        dx, dy, _, _ = self.grid_bbox(row=0, column=col)
        x, y, w, h = self._listboxes[col].bbox(row)
        return (int(x) + int(dx), int(y) + int(dy), int(w), int(h))

    def hide_column(self, col_index):
        """
        Hide the given column.  The column's state is still
        maintained: its values will still be returned by ``get()``, and
        you must supply its values when calling ``insert()``.  It is
        safe to call this on a column that is already hidden.

        :see: ``show_column()``
        """
        if self._labels:
            self._labels[col_index].grid_forget()
        self.listboxes[col_index].grid_forget()
        self.grid_columnconfigure(col_index, weight=0)

    def show_column(self, col_index):
        """
        Display a column that has been hidden using ``hide_column()``.
        It is safe to call this on a column that is not hidden.
        """
        weight = self._column_weights[col_index]
        if self._labels:
            self._labels[col_index].grid(column=col_index, row=0, sticky='news', padx=0, pady=0)
        self._listboxes[col_index].grid(column=col_index, row=1, sticky='news', padx=0, pady=0)
        self.grid_columnconfigure(col_index, weight=weight)

    def bind_to_labels(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Label`` widget in this
        mult-column listbox that will call ``func`` in response to the
        event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        return [label.bind(sequence, func, add) for label in self.column_labels]

    def bind_to_listboxes(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Listbox`` widget in this
        mult-column listbox that will call ``func`` in response to the
        event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        for listbox in self.listboxes:
            listbox.bind(sequence, func, add)

    def bind_to_columns(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Label`` and ``Tkinter.Listbox``
        widget in this mult-column listbox that will call ``func`` in
        response to the event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        return self.bind_to_labels(sequence, func, add) + self.bind_to_listboxes(sequence, func, add)

    def curselection(self, *args, **kwargs):
        return self._listboxes[0].curselection(*args, **kwargs)

    def selection_includes(self, *args, **kwargs):
        return self._listboxes[0].selection_includes(*args, **kwargs)

    def itemcget(self, *args, **kwargs):
        return self._listboxes[0].itemcget(*args, **kwargs)

    def size(self, *args, **kwargs):
        return self._listboxes[0].size(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self._listboxes[0].index(*args, **kwargs)

    def nearest(self, *args, **kwargs):
        return self._listboxes[0].nearest(*args, **kwargs)

    def activate(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.activate(*args, **kwargs)

    def delete(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.delete(*args, **kwargs)

    def scan_mark(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.scan_mark(*args, **kwargs)

    def scan_dragto(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.scan_dragto(*args, **kwargs)

    def see(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.see(*args, **kwargs)

    def selection_anchor(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_anchor(*args, **kwargs)

    def selection_clear(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_clear(*args, **kwargs)

    def selection_set(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_set(*args, **kwargs)

    def yview(self, *args, **kwargs):
        for lb in self._listboxes:
            v = lb.yview(*args, **kwargs)
        return v

    def yview_moveto(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.yview_moveto(*args, **kwargs)

    def yview_scroll(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.yview_scroll(*args, **kwargs)
    itemconfig = itemconfigure
    rowconfig = rowconfigure
    columnconfig = columnconfigure
    select_anchor = selection_anchor
    select_clear = selection_clear
    select_includes = selection_includes
    select_set = selection_set