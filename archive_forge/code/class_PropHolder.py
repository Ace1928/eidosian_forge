from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class PropHolder:
    """Base for property holders"""
    _attrMap = None

    def verify(self):
        """If the _attrMap attribute is not None, this
        checks all expected attributes are present; no
        unwanted attributes are present; and (if a
        checking function is found) checks each
        attribute has a valid value.  Either succeeds
        or raises an informative exception.
        """
        if self._attrMap is not None:
            for key in self.__dict__.keys():
                if key[0] != '_':
                    msg = 'Unexpected attribute %s found in %s' % (key, self)
                    assert key in self._attrMap, msg
            for attr, metavalue in self._attrMap.items():
                msg = 'Missing attribute %s from %s' % (attr, self)
                assert hasattr(self, attr), msg
                value = getattr(self, attr)
                args = (value, attr, self.__class__.__name__)
                assert metavalue.validate(value), 'Invalid value %s for attribute %s in class %s' % args
    if rl_config.shapeChecking:
        'This adds the ability to check every attribute assignment\n        as it is made. It slows down shapes but is a big help when\n        developing. It does not get defined if rl_config.shapeChecking = 0.\n        '

        def __setattr__(self, name, value):
            """By default we verify.  This could be off
            in some parallel base classes."""
            validateSetattr(self, name, value)

    def getProperties(self, recur=1):
        """Returns a list of all properties which can be edited and
        which are not marked as private. This may include 'child
        widgets' or 'primitive shapes'.  You are free to override
        this and provide alternative implementations; the default
        one simply returns everything without a leading underscore.
        """
        from reportlab.lib.validators import isValidChild
        props = {}
        for name in self.__dict__.keys():
            if name[0:1] != '_':
                component = getattr(self, name)
                if recur and isValidChild(component):
                    childProps = component.getProperties(recur=recur)
                    for childKey, childValue in childProps.items():
                        if childKey[0] == '[':
                            props['%s%s' % (name, childKey)] = childValue
                        else:
                            props['%s.%s' % (name, childKey)] = childValue
                else:
                    props[name] = component
        return props

    def setProperties(self, propDict):
        """Permits bulk setting of properties.  These may include
        child objects e.g. "chart.legend.width = 200".

        All assignments will be validated by the object as if they
        were set individually in python code.

        All properties of a top-level object are guaranteed to be
        set before any of the children, which may be helpful to
        widget designers.
        """
        childPropDicts = {}
        for name, value in propDict.items():
            parts = name.split('.', 1)
            if len(parts) == 1:
                setattr(self, name, value)
            else:
                childName, remains = parts
                try:
                    childPropDicts[childName][remains] = value
                except KeyError:
                    childPropDicts[childName] = {remains: value}
        for childName, childPropDict in childPropDicts.items():
            child = getattr(self, childName)
            child.setProperties(childPropDict)

    def dumpProperties(self, prefix=''):
        """Convenience. Lists them on standard output.  You
        may provide a prefix - mostly helps to generate code
        samples for documentation.
        """
        propList = list(self.getProperties().items())
        propList.sort()
        if prefix:
            prefix = prefix + '.'
        for name, value in propList:
            print('%s%s = %s' % (prefix, name, value))