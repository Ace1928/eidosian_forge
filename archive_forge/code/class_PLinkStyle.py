import sys, platform
from urllib.request import pathname2url
class PLinkStyle:
    """
    Provide platform specific thematic constants for use by Tk widgets.
    NOTE: A Tk interpreter must be created before instantiating an
    object in this class.
    """

    def __init__(self):
        self.ttk_style = ttk_style = ttk.Style()
        self.windowBG = ttk_style.lookup('Tframe', 'background')
        if sys.platform == 'darwin':
            self.font = 'Helvetica 16'
        else:
            self.font = 'Helvetica 12'