import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
class HelpMenu(Tk_.Menu):
    """Help Menu cascade.  Always contains the main SnapPy help entry.
    Additional help entries for specific tools, such as a Dirichlet
    viewer, may be added or removed.

    """

    def __init__(self, menubar):
        Tk_.Menu.__init__(self, menubar, name='help')
        if sys.platform != 'darwin':
            self.add_command(label='SnapPy Help ...', command=self.show_SnapPy_help)
        self.add_command(label=help_report_bugs_label, command=self.show_bugs_page)
        self.extra_commands = {}

    def show_SnapPy_help(self):
        open_html_docs('index.html')

    def show_bugs_page(self):
        open_html_docs('bugs.html')

    def extra_command(self, label, command):
        self.extra_commands[label] = command

    def activate(self, labels):
        """Manage extra help entries.
        Pass the labels of the extra commands to be activated.
        """
        end = self.index(Tk_.END)
        if sys.platform == 'darwin':
            self.delete(0, self.index(Tk_.END))
        elif end > 0:
            self.delete(1, self.index(Tk_.END))
        for label in labels:
            if label in self.extra_commands:
                self.add_command(label=label, command=self.extra_commands[label])