import base64
import json
import warnings
from binascii import hexlify
from collections import OrderedDict
from html import escape
from os import urandom
from pathlib import Path
from urllib.request import urlopen
from jinja2 import Environment, PackageLoader, Template
from .utilities import _camelify, _parse_size, none_max, none_min
class MacroElement(Element):
    """This is a parent class for Elements defined by a macro template.
    To compute your own element, all you have to do is:

    * To inherit from this class
    * Overwrite the '_name' attribute
    * Overwrite the '_template' attribute with something of the form::

        {% macro header(this, kwargs) %}
            ...
        {% endmacro %}

        {% macro html(this, kwargs) %}
            ...
        {% endmacro %}

        {% macro script(this, kwargs) %}
            ...
        {% endmacro %}

    """
    _template = Template('')

    def __init__(self):
        super().__init__()
        self._name = 'MacroElement'

    def render(self, **kwargs):
        """Renders the HTML representation of the element."""
        figure = self.get_root()
        assert isinstance(figure, Figure), 'You cannot render this Element if it is not in a Figure.'
        header = self._template.module.__dict__.get('header', None)
        if header is not None:
            figure.header.add_child(Element(header(self, kwargs)), name=self.get_name())
        html = self._template.module.__dict__.get('html', None)
        if html is not None:
            figure.html.add_child(Element(html(self, kwargs)), name=self.get_name())
        script = self._template.module.__dict__.get('script', None)
        if script is not None:
            figure.script.add_child(Element(script(self, kwargs)), name=self.get_name())
        for name, element in self._children.items():
            element.render(**kwargs)