from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
class PropertySet:
    defaults = {}

    def __init__(self, name, parent=None, **kw):
        """When initialized, it copies the class defaults;
        then takes a copy of the attributes of the parent
        if any.  All the work is done in init - styles
        should cost little to use at runtime."""
        assert 'name' not in self.defaults, "Class Defaults may not contain a 'name' attribute"
        assert 'parent' not in self.defaults, "Class Defaults may not contain a 'parent' attribute"
        if parent:
            assert parent.__class__ == self.__class__, 'Parent style %s must have same class as new style %s' % (parent.__class__.__name__, self.__class__.__name__)
        self.name = name
        self.parent = parent
        self.__dict__.update(self.defaults)
        self.refresh()
        self._setKwds(**kw)

    def _setKwds(self, **kw):
        for key, value in kw.items():
            self.__dict__[key] = value

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self.name)

    def refresh(self):
        """re-fetches attributes from the parent on demand;
        use if you have been hacking the styles.  This is
        used by __init__"""
        if self.parent:
            for key, value in self.parent.__dict__.items():
                if key not in ['name', 'parent']:
                    self.__dict__[key] = value

    def listAttrs(self, indent=''):
        print(indent + 'name =', self.name)
        print(indent + 'parent =', self.parent)
        keylist = list(self.__dict__.keys())
        keylist.sort()
        keylist.remove('name')
        keylist.remove('parent')
        for key in keylist:
            value = self.__dict__.get(key, None)
            print(indent + '%s = %s' % (key, value))

    def clone(self, name, parent=None, **kwds):
        r = self.__class__(name, parent)
        r.__dict__ = self.__dict__.copy()
        r.name = name
        r.parent = parent is None and self or parent
        r._setKwds(**kwds)
        return r