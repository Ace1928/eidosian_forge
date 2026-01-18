import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
class DownloaderGUI:
    """
    Graphical interface for downloading packages from the NLTK data
    server.
    """
    COLUMNS = ['', 'Identifier', 'Name', 'Size', 'Status', 'Unzipped Size', 'Copyright', 'Contact', 'License', 'Author', 'Subdir', 'Checksum']
    'A list of the names of columns.  This controls the order in\n       which the columns will appear.  If this is edited, then\n       ``_package_to_columns()`` may need to be edited to match.'
    COLUMN_WEIGHTS = {'': 0, 'Name': 5, 'Size': 0, 'Status': 0}
    'A dictionary specifying how columns should be resized when the\n       table is resized.  Columns with weight 0 will not be resized at\n       all; and columns with high weight will be resized more.\n       Default weight (for columns not explicitly listed) is 1.'
    COLUMN_WIDTHS = {'': 1, 'Identifier': 20, 'Name': 45, 'Size': 10, 'Unzipped Size': 10, 'Status': 12}
    'A dictionary specifying how wide each column should be, in\n       characters.  The default width (for columns not explicitly\n       listed) is specified by ``DEFAULT_COLUMN_WIDTH``.'
    DEFAULT_COLUMN_WIDTH = 30
    'The default width for columns that are not explicitly listed\n       in ``COLUMN_WIDTHS``.'
    INITIAL_COLUMNS = ['', 'Identifier', 'Name', 'Size', 'Status']
    'The set of columns that should be displayed by default.'
    for c in COLUMN_WEIGHTS:
        assert c in COLUMNS
    for c in COLUMN_WIDTHS:
        assert c in COLUMNS
    for c in INITIAL_COLUMNS:
        assert c in COLUMNS
    _BACKDROP_COLOR = ('#000', '#ccc')
    _ROW_COLOR = {Downloader.INSTALLED: ('#afa', '#080'), Downloader.PARTIAL: ('#ffa', '#880'), Downloader.STALE: ('#faa', '#800'), Downloader.NOT_INSTALLED: ('#fff', '#888')}
    _MARK_COLOR = ('#000', '#ccc')
    _FRONT_TAB_COLOR = ('#fff', '#45c')
    _BACK_TAB_COLOR = ('#aaa', '#67a')
    _PROGRESS_COLOR = ('#f00', '#aaa')
    _TAB_FONT = 'helvetica -16 bold'

    def __init__(self, dataserver, use_threads=True):
        self._ds = dataserver
        self._use_threads = use_threads
        self._download_lock = threading.Lock()
        self._download_msg_queue = []
        self._download_abort_queue = []
        self._downloading = False
        self._afterid = {}
        self._log_messages = []
        self._log_indent = 0
        self._log('NLTK Downloader Started!')
        top = self.top = Tk()
        top.geometry('+50+50')
        top.title('NLTK Downloader')
        top.configure(background=self._BACKDROP_COLOR[1])
        top.bind('<Control-q>', self.destroy)
        top.bind('<Control-x>', self.destroy)
        self._destroyed = False
        self._column_vars = {}
        self._init_widgets()
        self._init_menu()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror('Error reading from server', e)
        except URLError as e:
            showerror('Error connecting to server', e.reason)
        self._show_info()
        self._select_columns()
        self._table.select(0)
        self._table.bind('<Destroy>', self._destroy)

    def _log(self, msg):
        self._log_messages.append('{} {}{}'.format(time.ctime(), ' | ' * self._log_indent, msg))

    def _init_widgets(self):
        f1 = Frame(self.top, relief='raised', border=2, padx=8, pady=0)
        f1.pack(sid='top', expand=True, fill='both')
        f1.grid_rowconfigure(2, weight=1)
        f1.grid_columnconfigure(0, weight=1)
        Frame(f1, height=8).grid(column=0, row=0)
        tabframe = Frame(f1)
        tabframe.grid(column=0, row=1, sticky='news')
        tableframe = Frame(f1)
        tableframe.grid(column=0, row=2, sticky='news')
        buttonframe = Frame(f1)
        buttonframe.grid(column=0, row=3, sticky='news')
        Frame(f1, height=8).grid(column=0, row=4)
        infoframe = Frame(f1)
        infoframe.grid(column=0, row=5, sticky='news')
        Frame(f1, height=8).grid(column=0, row=6)
        progressframe = Frame(self.top, padx=3, pady=3, background=self._BACKDROP_COLOR[1])
        progressframe.pack(side='bottom', fill='x')
        self.top['border'] = 0
        self.top['highlightthickness'] = 0
        self._tab_names = ['Collections', 'Corpora', 'Models', 'All Packages']
        self._tabs = {}
        for i, tab in enumerate(self._tab_names):
            label = Label(tabframe, text=tab, font=self._TAB_FONT)
            label.pack(side='left', padx=(i + 1) % 2 * 10)
            label.bind('<Button-1>', self._select_tab)
            self._tabs[tab.lower()] = label
        column_weights = [self.COLUMN_WEIGHTS.get(column, 1) for column in self.COLUMNS]
        self._table = Table(tableframe, self.COLUMNS, column_weights=column_weights, highlightthickness=0, listbox_height=16, reprfunc=self._table_reprfunc)
        self._table.columnconfig(0, foreground=self._MARK_COLOR[0])
        for i, column in enumerate(self.COLUMNS):
            width = self.COLUMN_WIDTHS.get(column, self.DEFAULT_COLUMN_WIDTH)
            self._table.columnconfig(i, width=width)
        self._table.pack(expand=True, fill='both')
        self._table.focus()
        self._table.bind_to_listboxes('<Double-Button-1>', self._download)
        self._table.bind('<space>', self._table_mark)
        self._table.bind('<Return>', self._download)
        self._table.bind('<Left>', self._prev_tab)
        self._table.bind('<Right>', self._next_tab)
        self._table.bind('<Control-a>', self._mark_all)
        infoframe.grid_columnconfigure(1, weight=1)
        info = [('url', 'Server Index:', self._set_url), ('download_dir', 'Download Directory:', self._set_download_dir)]
        self._info = {}
        for i, (key, label, callback) in enumerate(info):
            Label(infoframe, text=label).grid(column=0, row=i, sticky='e')
            entry = Entry(infoframe, font='courier', relief='groove', disabledforeground='#007aff', foreground='#007aff')
            self._info[key] = (entry, callback)
            entry.bind('<Return>', self._info_save)
            entry.bind('<Button-1>', lambda e, key=key: self._info_edit(key))
            entry.grid(column=1, row=i, sticky='ew')
        self.top.bind('<Button-1>', self._info_save)
        self._download_button = Button(buttonframe, text='Download', command=self._download, width=8)
        self._download_button.pack(side='left')
        self._refresh_button = Button(buttonframe, text='Refresh', command=self._refresh, width=8)
        self._refresh_button.pack(side='right')
        self._progresslabel = Label(progressframe, text='', foreground=self._BACKDROP_COLOR[0], background=self._BACKDROP_COLOR[1])
        self._progressbar = Canvas(progressframe, width=200, height=16, background=self._PROGRESS_COLOR[1], relief='sunken', border=1)
        self._init_progressbar()
        self._progressbar.pack(side='right')
        self._progresslabel.pack(side='left')

    def _init_menu(self):
        menubar = Menu(self.top)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label='Download', underline=0, command=self._download, accelerator='Return')
        filemenu.add_separator()
        filemenu.add_command(label='Change Server Index', underline=7, command=lambda: self._info_edit('url'))
        filemenu.add_command(label='Change Download Directory', underline=0, command=lambda: self._info_edit('download_dir'))
        filemenu.add_separator()
        filemenu.add_command(label='Show Log', underline=5, command=self._show_log)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
        menubar.add_cascade(label='File', underline=0, menu=filemenu)
        viewmenu = Menu(menubar, tearoff=0)
        for column in self._table.column_names[2:]:
            var = IntVar(self.top)
            assert column not in self._column_vars
            self._column_vars[column] = var
            if column in self.INITIAL_COLUMNS:
                var.set(1)
            viewmenu.add_checkbutton(label=column, underline=0, variable=var, command=self._select_columns)
        menubar.add_cascade(label='View', underline=0, menu=viewmenu)
        sortmenu = Menu(menubar, tearoff=0)
        for column in self._table.column_names[1:]:
            sortmenu.add_command(label='Sort by %s' % column, command=lambda c=column: self._table.sort_by(c, 'ascending'))
        sortmenu.add_separator()
        for column in self._table.column_names[1:]:
            sortmenu.add_command(label='Reverse sort by %s' % column, command=lambda c=column: self._table.sort_by(c, 'descending'))
        menubar.add_cascade(label='Sort', underline=0, menu=sortmenu)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label='About', underline=0, command=self.about)
        helpmenu.add_command(label='Instructions', underline=0, command=self.help, accelerator='F1')
        menubar.add_cascade(label='Help', underline=0, menu=helpmenu)
        self.top.bind('<F1>', self.help)
        self.top.config(menu=menubar)

    def _select_columns(self):
        for column, var in self._column_vars.items():
            if var.get():
                self._table.show_column(column)
            else:
                self._table.hide_column(column)

    def _refresh(self):
        self._ds.clear_status_cache()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror('Error reading from server', e)
        except URLError as e:
            showerror('Error connecting to server', e.reason)
        self._table.select(0)

    def _info_edit(self, info_key):
        self._info_save()
        entry, callback = self._info[info_key]
        entry['state'] = 'normal'
        entry['relief'] = 'sunken'
        entry.focus()

    def _info_save(self, e=None):
        focus = self._table
        for entry, callback in self._info.values():
            if entry['state'] == 'disabled':
                continue
            if e is not None and e.widget is entry and (e.keysym != 'Return'):
                focus = entry
            else:
                entry['state'] = 'disabled'
                entry['relief'] = 'groove'
                callback(entry.get())
        focus.focus()

    def _table_reprfunc(self, row, col, val):
        if self._table.column_names[col].endswith('Size'):
            if isinstance(val, str):
                return '  %s' % val
            elif val < 1024 ** 2:
                return '  %.1f KB' % (val / 1024.0 ** 1)
            elif val < 1024 ** 3:
                return '  %.1f MB' % (val / 1024.0 ** 2)
            else:
                return '  %.1f GB' % (val / 1024.0 ** 3)
        if col in (0, ''):
            return str(val)
        else:
            return '  %s' % val

    def _set_url(self, url):
        if url == self._ds.url:
            return
        try:
            self._ds.url = url
            self._fill_table()
        except OSError as e:
            showerror('Error Setting Server Index', str(e))
        self._show_info()

    def _set_download_dir(self, download_dir):
        if self._ds.download_dir == download_dir:
            return
        self._ds.download_dir = download_dir
        try:
            self._fill_table()
        except HTTPError as e:
            showerror('Error reading from server', e)
        except URLError as e:
            showerror('Error connecting to server', e.reason)
        self._show_info()

    def _show_info(self):
        print('showing info', self._ds.url)
        for entry, cb in self._info.values():
            entry['state'] = 'normal'
            entry.delete(0, 'end')
        self._info['url'][0].insert(0, self._ds.url)
        self._info['download_dir'][0].insert(0, self._ds.download_dir)
        for entry, cb in self._info.values():
            entry['state'] = 'disabled'

    def _prev_tab(self, *e):
        for i, tab in enumerate(self._tab_names):
            if tab.lower() == self._tab and i > 0:
                self._tab = self._tab_names[i - 1].lower()
                try:
                    return self._fill_table()
                except HTTPError as e:
                    showerror('Error reading from server', e)
                except URLError as e:
                    showerror('Error connecting to server', e.reason)

    def _next_tab(self, *e):
        for i, tab in enumerate(self._tab_names):
            if tab.lower() == self._tab and i < len(self._tabs) - 1:
                self._tab = self._tab_names[i + 1].lower()
                try:
                    return self._fill_table()
                except HTTPError as e:
                    showerror('Error reading from server', e)
                except URLError as e:
                    showerror('Error connecting to server', e.reason)

    def _select_tab(self, event):
        self._tab = event.widget['text'].lower()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror('Error reading from server', e)
        except URLError as e:
            showerror('Error connecting to server', e.reason)
    _tab = 'collections'
    _rows = None

    def _fill_table(self):
        selected_row = self._table.selected_row()
        self._table.clear()
        if self._tab == 'all packages':
            items = self._ds.packages()
        elif self._tab == 'corpora':
            items = self._ds.corpora()
        elif self._tab == 'models':
            items = self._ds.models()
        elif self._tab == 'collections':
            items = self._ds.collections()
        else:
            assert 0, 'bad tab value %r' % self._tab
        rows = [self._package_to_columns(item) for item in items]
        self._table.extend(rows)
        for tab, label in self._tabs.items():
            if tab == self._tab:
                label.configure(foreground=self._FRONT_TAB_COLOR[0], background=self._FRONT_TAB_COLOR[1])
            else:
                label.configure(foreground=self._BACK_TAB_COLOR[0], background=self._BACK_TAB_COLOR[1])
        self._table.sort_by('Identifier', order='ascending')
        self._color_table()
        self._table.select(selected_row)
        self.top.after(150, self._table._scrollbar.set, *self._table._mlb.yview())
        self.top.after(300, self._table._scrollbar.set, *self._table._mlb.yview())

    def _update_table_status(self):
        for row_num in range(len(self._table)):
            status = self._ds.status(self._table[row_num, 'Identifier'])
            self._table[row_num, 'Status'] = status
        self._color_table()

    def _download(self, *e):
        if self._use_threads:
            return self._download_threaded(*e)
        marked = [self._table[row, 'Identifier'] for row in range(len(self._table)) if self._table[row, 0] != '']
        selection = self._table.selected_row()
        if not marked and selection is not None:
            marked = [self._table[selection, 'Identifier']]
        download_iter = self._ds.incr_download(marked, self._ds.download_dir)
        self._log_indent = 0
        self._download_cb(download_iter, marked)
    _DL_DELAY = 10

    def _download_cb(self, download_iter, ids):
        try:
            msg = next(download_iter)
        except StopIteration:
            self._update_table_status()
            afterid = self.top.after(10, self._show_progress, 0)
            self._afterid['_download_cb'] = afterid
            return

        def show(s):
            self._progresslabel['text'] = s
            self._log(s)
        if isinstance(msg, ProgressMessage):
            self._show_progress(msg.progress)
        elif isinstance(msg, ErrorMessage):
            show(msg.message)
            if msg.package is not None:
                self._select(msg.package.id)
            self._show_progress(None)
            return
        elif isinstance(msg, StartCollectionMessage):
            show('Downloading collection %s' % msg.collection.id)
            self._log_indent += 1
        elif isinstance(msg, StartPackageMessage):
            show('Downloading package %s' % msg.package.id)
        elif isinstance(msg, UpToDateMessage):
            show('Package %s is up-to-date!' % msg.package.id)
        elif isinstance(msg, FinishDownloadMessage):
            show('Finished downloading %r.' % msg.package.id)
        elif isinstance(msg, StartUnzipMessage):
            show('Unzipping %s' % msg.package.filename)
        elif isinstance(msg, FinishCollectionMessage):
            self._log_indent -= 1
            show('Finished downloading collection %r.' % msg.collection.id)
            self._clear_mark(msg.collection.id)
        elif isinstance(msg, FinishPackageMessage):
            self._clear_mark(msg.package.id)
        afterid = self.top.after(self._DL_DELAY, self._download_cb, download_iter, ids)
        self._afterid['_download_cb'] = afterid

    def _select(self, id):
        for row in range(len(self._table)):
            if self._table[row, 'Identifier'] == id:
                self._table.select(row)
                return

    def _color_table(self):
        for row in range(len(self._table)):
            bg, sbg = self._ROW_COLOR[self._table[row, 'Status']]
            fg, sfg = ('black', 'white')
            self._table.rowconfig(row, foreground=fg, selectforeground=sfg, background=bg, selectbackground=sbg)
            self._table.itemconfigure(row, 0, foreground=self._MARK_COLOR[0], background=self._MARK_COLOR[1])

    def _clear_mark(self, id):
        for row in range(len(self._table)):
            if self._table[row, 'Identifier'] == id:
                self._table[row, 0] = ''

    def _mark_all(self, *e):
        for row in range(len(self._table)):
            self._table[row, 0] = 'X'

    def _table_mark(self, *e):
        selection = self._table.selected_row()
        if selection >= 0:
            if self._table[selection][0] != '':
                self._table[selection, 0] = ''
            else:
                self._table[selection, 0] = 'X'
        self._table.select(delta=1)

    def _show_log(self):
        text = '\n'.join(self._log_messages)
        ShowText(self.top, 'NLTK Downloader Log', text)

    def _package_to_columns(self, pkg):
        """
        Given a package, return a list of values describing that
        package, one for each column in ``self.COLUMNS``.
        """
        row = []
        for column_index, column_name in enumerate(self.COLUMNS):
            if column_index == 0:
                row.append('')
            elif column_name == 'Identifier':
                row.append(pkg.id)
            elif column_name == 'Status':
                row.append(self._ds.status(pkg))
            else:
                attr = column_name.lower().replace(' ', '_')
                row.append(getattr(pkg, attr, 'n/a'))
        return row

    def destroy(self, *e):
        if self._destroyed:
            return
        self.top.destroy()
        self._destroyed = True

    def _destroy(self, *e):
        if self.top is not None:
            for afterid in self._afterid.values():
                self.top.after_cancel(afterid)
        if self._downloading and self._use_threads:
            self._abort_download()
        self._column_vars.clear()

    def mainloop(self, *args, **kwargs):
        self.top.mainloop(*args, **kwargs)
    HELP = textwrap.dedent('    This tool can be used to download a variety of corpora and models\n    that can be used with NLTK.  Each corpus or model is distributed\n    in a single zip file, known as a "package file."  You can\n    download packages individually, or you can download pre-defined\n    collections of packages.\n\n    When you download a package, it will be saved to the "download\n    directory."  A default download directory is chosen when you run\n\n    the downloader; but you may also select a different download\n    directory.  On Windows, the default download directory is\n\n\n    "package."\n\n    The NLTK downloader can be used to download a variety of corpora,\n    models, and other data packages.\n\n    Keyboard shortcuts::\n      [return]\t Download\n      [up]\t Select previous package\n      [down]\t Select next package\n      [left]\t Select previous tab\n      [right]\t Select next tab\n    ')

    def help(self, *e):
        try:
            ShowText(self.top, 'Help: NLTK Downloader', self.HELP.strip(), width=75, font='fixed')
        except:
            ShowText(self.top, 'Help: NLTK Downloader', self.HELP.strip(), width=75)

    def about(self, *e):
        ABOUT = 'NLTK Downloader\n' + 'Written by Edward Loper'
        TITLE = 'About: NLTK Downloader'
        try:
            from tkinter.messagebox import Message
            Message(message=ABOUT, title=TITLE).show()
        except ImportError:
            ShowText(self.top, TITLE, ABOUT)
    _gradient_width = 5

    def _init_progressbar(self):
        c = self._progressbar
        width, height = (int(c['width']), int(c['height']))
        for i in range(0, int(c['width']) * 2 // self._gradient_width):
            c.create_line(i * self._gradient_width + 20, -20, i * self._gradient_width - height - 20, height + 20, width=self._gradient_width, fill='#%02x0000' % (80 + abs(i % 6 - 3) * 12))
        c.addtag_all('gradient')
        c.itemconfig('gradient', state='hidden')
        c.addtag_withtag('redbox', c.create_rectangle(0, 0, 0, 0, fill=self._PROGRESS_COLOR[0]))

    def _show_progress(self, percent):
        c = self._progressbar
        if percent is None:
            c.coords('redbox', 0, 0, 0, 0)
            c.itemconfig('gradient', state='hidden')
        else:
            width, height = (int(c['width']), int(c['height']))
            x = percent * int(width) // 100 + 1
            c.coords('redbox', 0, 0, x, height + 1)

    def _progress_alive(self):
        c = self._progressbar
        if not self._downloading:
            c.itemconfig('gradient', state='hidden')
        else:
            c.itemconfig('gradient', state='normal')
            x1, y1, x2, y2 = c.bbox('gradient')
            if x1 <= -100:
                c.move('gradient', self._gradient_width * 6 - 4, 0)
            else:
                c.move('gradient', -4, 0)
            afterid = self.top.after(200, self._progress_alive)
            self._afterid['_progress_alive'] = afterid

    def _download_threaded(self, *e):
        if self._downloading:
            self._abort_download()
            return
        self._download_button['text'] = 'Cancel'
        marked = [self._table[row, 'Identifier'] for row in range(len(self._table)) if self._table[row, 0] != '']
        selection = self._table.selected_row()
        if not marked and selection is not None:
            marked = [self._table[selection, 'Identifier']]
        ds = Downloader(self._ds.url, self._ds.download_dir)
        assert self._download_msg_queue == []
        assert self._download_abort_queue == []
        self._DownloadThread(ds, marked, self._download_lock, self._download_msg_queue, self._download_abort_queue).start()
        self._log_indent = 0
        self._downloading = True
        self._monitor_message_queue()
        self._progress_alive()

    def _abort_download(self):
        if self._downloading:
            self._download_lock.acquire()
            self._download_abort_queue.append('abort')
            self._download_lock.release()

    class _DownloadThread(threading.Thread):

        def __init__(self, data_server, items, lock, message_queue, abort):
            self.data_server = data_server
            self.items = items
            self.lock = lock
            self.message_queue = message_queue
            self.abort = abort
            threading.Thread.__init__(self)

        def run(self):
            for msg in self.data_server.incr_download(self.items):
                self.lock.acquire()
                self.message_queue.append(msg)
                if self.abort:
                    self.message_queue.append('aborted')
                    self.lock.release()
                    return
                self.lock.release()
            self.lock.acquire()
            self.message_queue.append('finished')
            self.lock.release()
    _MONITOR_QUEUE_DELAY = 100

    def _monitor_message_queue(self):

        def show(s):
            self._progresslabel['text'] = s
            self._log(s)
        if not self._download_lock.acquire():
            return
        for msg in self._download_msg_queue:
            if msg == 'finished' or msg == 'aborted':
                self._update_table_status()
                self._downloading = False
                self._download_button['text'] = 'Download'
                del self._download_msg_queue[:]
                del self._download_abort_queue[:]
                self._download_lock.release()
                if msg == 'aborted':
                    show('Download aborted!')
                    self._show_progress(None)
                else:
                    afterid = self.top.after(100, self._show_progress, None)
                    self._afterid['_monitor_message_queue'] = afterid
                return
            elif isinstance(msg, ProgressMessage):
                self._show_progress(msg.progress)
            elif isinstance(msg, ErrorMessage):
                show(msg.message)
                if msg.package is not None:
                    self._select(msg.package.id)
                self._show_progress(None)
                self._downloading = False
                return
            elif isinstance(msg, StartCollectionMessage):
                show('Downloading collection %r' % msg.collection.id)
                self._log_indent += 1
            elif isinstance(msg, StartPackageMessage):
                self._ds.clear_status_cache(msg.package.id)
                show('Downloading package %r' % msg.package.id)
            elif isinstance(msg, UpToDateMessage):
                show('Package %s is up-to-date!' % msg.package.id)
            elif isinstance(msg, FinishDownloadMessage):
                show('Finished downloading %r.' % msg.package.id)
            elif isinstance(msg, StartUnzipMessage):
                show('Unzipping %s' % msg.package.filename)
            elif isinstance(msg, FinishUnzipMessage):
                show('Finished installing %s' % msg.package.id)
            elif isinstance(msg, FinishCollectionMessage):
                self._log_indent -= 1
                show('Finished downloading collection %r.' % msg.collection.id)
                self._clear_mark(msg.collection.id)
            elif isinstance(msg, FinishPackageMessage):
                self._update_table_status()
                self._clear_mark(msg.package.id)
        if self._download_abort_queue:
            self._progresslabel['text'] = 'Aborting download...'
        del self._download_msg_queue[:]
        self._download_lock.release()
        afterid = self.top.after(self._MONITOR_QUEUE_DELAY, self._monitor_message_queue)
        self._afterid['_monitor_message_queue'] = afterid