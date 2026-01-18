from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
class StyleSheet1:
    """
    This may or may not be used.  The idea is to:
    
    1. slightly simplify construction of stylesheets;
    
    2. enforce rules to validate styles when added
       (e.g. we may choose to disallow having both
       'heading1' and 'Heading1' - actual rules are
       open to discussion);
       
    3. allow aliases and alternate style lookup
       mechanisms
       
    4. Have a place to hang style-manipulation
       methods (save, load, maybe support a GUI
       editor)
   
    Access is via getitem, so they can be
    compatible with plain old dictionaries.
    """

    def __init__(self):
        self.byName = {}
        self.byAlias = {}

    def __getitem__(self, key):
        try:
            return self.byAlias[key]
        except KeyError:
            try:
                return self.byName[key]
            except KeyError:
                raise KeyError("Style '%s' not found in stylesheet" % key)

    def get(self, key, default=_stylesheet1_undefined):
        try:
            return self[key]
        except KeyError:
            if default != _stylesheet1_undefined:
                return default
            raise

    def __contains__(self, key):
        return key in self.byAlias or key in self.byName

    def has_key(self, key):
        return key in self

    def add(self, style, alias=None):
        key = style.name
        if key in self.byName:
            raise KeyError("Style '%s' already defined in stylesheet" % key)
        if key in self.byAlias:
            raise KeyError("Style name '%s' is already an alias in stylesheet" % key)
        if alias:
            if alias in self.byName:
                raise KeyError("Style '%s' already defined in stylesheet" % alias)
            if alias in self.byAlias:
                raise KeyError("Alias name '%s' is already an alias in stylesheet" % alias)
        self.byName[key] = style
        if alias:
            self.byAlias[alias] = style

    def list(self):
        styles = list(self.byName.items())
        styles.sort()
        alii = {}
        for alias, style in list(self.byAlias.items()):
            alii[style] = alias
        for name, style in styles:
            alias = alii.get(style, None)
            print(name, alias)
            style.listAttrs('    ')
            print()