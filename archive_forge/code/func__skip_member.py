import re
from sphinx.ext.napoleon import (
from ... import __version__
from ...interfaces.base import BaseInterface, TraitedSpec
from .docstring import NipypeDocstring, InterfaceDocstring
def _skip_member(app, what, name, obj, skip, options):
    """
    Determine if private and special class members are included in docs.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process
    what : str
        A string specifying the type of the object to which the member
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : str
        The name of the member.
    obj : module, class, exception, function, method, or attribute.
        For example, if the member is the __init__ method of class A, then
        `obj` will be `A.__init__`.
    skip : bool
        A boolean indicating if autodoc will skip this member if `_skip_member`
        does not override the decision
    options : sphinx.ext.autodoc.Options
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.
    Returns
    -------
    bool
        True if the member should be skipped during creation of the docs,
        False if it should be included in the docs.

    """
    patterns = [pat if hasattr(pat, 'search') else re.compile(pat) for pat in app.config.nipype_skip_classes]
    isbase = False
    try:
        isbase = issubclass(obj, BaseInterface)
        if issubclass(obj, TraitedSpec):
            return True
    except TypeError:
        pass
    if isbase:
        for pattern in patterns:
            if pattern.search(name):
                return True
    return _napoleon_skip_member(app, what, name, obj, skip, options)