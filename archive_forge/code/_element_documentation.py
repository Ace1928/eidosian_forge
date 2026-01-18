import itertools
from typing import (
from zope.interface import implementer
from twisted.web.error import (
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader

        Implement L{IRenderable} to allow one L{Element} to be embedded in
        another's template or rendering output.

        (This will simply load the template from the C{loader}; when used in a
        template, the flattening engine will keep track of this object
        separately as the object to lookup renderers on and call
        L{Element.renderer} to look them up.  The resulting object from this
        method is not directly associated with this L{Element}.)
        