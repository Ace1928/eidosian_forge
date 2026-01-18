import os
import sys
import re
import time
from collections.abc import Mapping  # Python 3.5 or newer
from IPython.core.displayhook import DisplayHook
from tkinter.messagebox import askyesno
from .gui import *
from . import filedialog
from .exceptions import SnapPeaFatalError
from .app_menus import HelpMenu, EditMenu, WindowMenu, ListedWindow
from .app_menus import dirichlet_menus, horoball_menus, inside_view_menus, plink_menus
from .app_menus import add_menu, scut, open_html_docs
from .browser import Browser
from .horoviewer import HoroballViewer
from .infowindow import about_snappy, InfoWindow
from .polyviewer import PolyhedronViewer
from .raytracing.inside_viewer import InsideViewer
from .settings import Settings, SettingsDialog
from .phone_home import update_needed
from .SnapPy import SnapPea_interrupt, msg_stream
from .shell import SnapPyInteractiveShellEmbed
from .tkterminal import TkTerm, snappy_path
from plink import LinkEditor
from plink.smooth import Smoother
import site
import pydoc
class SnapPyTerm(TkTerm, ListedWindow):
    """
    The main window of the SnapPy app, which runs an embedded IPython shell.
    """

    def __init__(self):
        self.ipython_shell = shell = SnapPyInteractiveShellEmbed.instance(banner1=app_banner + update_needed())
        shell.output = self
        shell.set_hook('show_in_pager', IPython_pager)
        self.main_window = self
        self.menu_title = 'SnapPy Shell'
        self.register_window(self)
        TkTerm.__init__(self, shell, name='SnapPy Command Shell')
        self.settings = SnapPySettings(self)
        self.start_interaction()
        if sys.platform == 'darwin':
            assert str(self.window) == '.'
            self.window.protocol('WM_DELETE_WINDOW', lambda: self.window.iconify())
            self.window.createcommand('::tk::mac::OpenDocument', self.OSX_open_filelist)
        else:
            self.window.tk.call('namespace', 'import', '::tk::dialog::file::')
            self.window.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
            self.window.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
        self.encoding = None

    def add_bindings(self):
        self.window.bind('<<Paste>>', self.edit_paste)

    def about_window(self):
        window = self.window
        if not hasattr(window, 'about_snappy'):
            window.about_snappy = about_snappy(window)
        else:
            window.about_snappy.deiconify()
            window.about_snappy.lift()
            window.about_snappy.focus_force()

    def build_menus(self):
        window = self.window
        self.menubar = menubar = Tk_.Menu(window)
        Python_menu = Tk_.Menu(menubar, name='apple')
        Python_menu.add_command(label='About SnapPy...', command=self.about_window)
        if sys.platform == 'darwin':
            window.createcommand('::tk::mac::ShowPreferences', self.edit_settings)
            window.createcommand('::tk::mac::Quit', self.close)
        else:
            Python_menu.add_separator()
            Python_menu.add_command(label='Settings...', command=self.edit_settings)
            Python_menu.add_separator()
            Python_menu.add_command(label='Quit SnapPy', command=self.close)
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
        File_menu = Tk_.Menu(menubar, name='file')
        add_menu(window, File_menu, 'Open...', self.open_file)
        add_menu(window, File_menu, 'Open link...', self.open_link_file)
        add_menu(window, File_menu, 'Save', self.save_file, state='disabled')
        add_menu(window, File_menu, 'Save as...', self.save_file_as)
        menubar.add_cascade(label='File', menu=File_menu)
        menubar.add_cascade(label='Edit ', menu=EditMenu(menubar, self.edit_actions))
        if sys.platform == 'darwin':
            menubar.add_cascade(label='View', menu=Tk_.Menu(menubar, name='view'))
        menubar.add_cascade(label='Window', menu=WindowMenu(menubar))
        help_menu = HelpMenu(menubar)
        if sys.platform == 'darwin':
            window.createcommand('::tk::mac::ShowHelp', help_menu.show_SnapPy_help)
        menubar.add_cascade(label='Help', menu=help_menu)

    def edit_settings(self):
        terminal.can_quit = False
        if sys.platform == 'darwin':
            self.window.deletecommand('::tk::mac::ShowPreferences')
        else:
            apple_menu = self.menubar.children['apple']
            apple_menu.entryconfig(2, state='disabled')
        dialog = SettingsDialog(self.window, self.settings)
        terminal.add_blocker(dialog, 'Changes to your settings will be lost if you quit SnapPy now.')
        dialog.run()
        terminal.remove_blocker(dialog)
        if dialog.okay:
            answer = askyesno('Save?', 'Do you want to save these settings?')
            if answer:
                self.settings.write_settings()
        if sys.platform == 'darwin':
            self.window.createcommand('::tk::mac::ShowPreferences', self.edit_settings)
        else:
            apple_menu.entryconfig(2, state='active')
        self.can_quit = True

    def OSX_open_filelist(self, *args):
        for arg in args:
            sys.stderr.write(repr(arg) + '\n')

    def open_file(self, event=None):
        openfile = filedialog.askopenfile(parent=self.window, title='Run Saved Transcript In Current Namespace', defaultextension='.py', filetypes=[('Python and text files', '*.py *.ipy *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
        if openfile:
            lines = openfile.readlines()
            openfile.close()
            if re.search('%\\s*([vV]irtual)*\\s*[lL]ink\\s*[Pp]rojection', lines[0]):
                tkMessageBox.showwarning('Bad file', 'This is a SnapPea link projection file, not a session transcript.')
            elif re.search('%\\s*[tT]riangulation', lines[0]):
                tkMessageBox.showwarning('Bad file', 'This is a SnapPea triangulation file, not a session transcript.')
            elif re.search('%\\s*Generators', lines[0]):
                tkMessageBox.showwarning('Bad file', 'This is a SnapPea generator file, not a session transcript.')
            else:
                while lines[0][0] in ('#', '\n'):
                    lines.pop(0)
                for line in lines:
                    if line.startswith('#'):
                        continue
                    self.write(line[:-1].lstrip(), mark=Tk_.INSERT, advance=False)
                    self.handle_return(event=None)

    def open_link_file(self, event=None):
        openfile = filedialog.askopenfile(title='Load Link Projection File', defaultextension='.lnk', filetypes=[('Link and text files', '*.lnk *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
        if openfile:
            if not re.search('%\\s*([vV]irtual)*\\s*[lL]ink\\s*[Pp]rojection', openfile.readline()):
                tkMessageBox.showwarning('Bad file', 'This is not a SnapPea link projection file')
                openfile.close()
            else:
                name = openfile.name
                openfile.close()
                line = 'Manifold()\n'
                self.write(line)
                self.interact_handle_input(line)
                self.interact_prompt()
                M = self.IP.user_ns['_']
                M.LE.load(file_name=name)

    def save_file_as(self, event=None):
        savefile = filedialog.asksaveasfile(parent=self.window, mode='w', title='Save Transcript as a Python script', defaultextension='.py', filetypes=[('Python and text files', '*.py *.ipy *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
        if savefile:
            savefile.write('#!/usr/bin/env/python\n# This script was saved by SnapPy on %s.\n' % time.asctime())
            inputs = self.IP.history_manager.input_hist_raw
            results = self.IP.history_manager.output_hist
            for n in range(1, len(inputs)):
                savefile.write('\n' + re.sub('\n+', '\n', inputs[n]) + '\n')
                try:
                    output = repr(results[n]).split('\n')
                except:
                    continue
                for line in output:
                    savefile.write('#' + line + '\n')
            savefile.close()

    def save_file(self, event=None):
        self.window.bell()
        self.write2('Save As\n')