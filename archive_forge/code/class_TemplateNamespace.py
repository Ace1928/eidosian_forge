import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class TemplateNamespace(Namespace):
    """A :class:`.Namespace` specific to a :class:`.Template` instance."""

    def __init__(self, name, context, template=None, templateuri=None, callables=None, inherits=None, populate_self=True, calling_uri=None):
        self.name = name
        self.context = context
        self.inherits = inherits
        if callables is not None:
            self.callables = {c.__name__: c for c in callables}
        if templateuri is not None:
            self.template = _lookup_template(context, templateuri, calling_uri)
            self._templateuri = self.template.module._template_uri
        elif template is not None:
            self.template = template
            self._templateuri = template.module._template_uri
        else:
            raise TypeError("'template' argument is required.")
        if populate_self:
            lclcallable, lclcontext = _populate_self_namespace(context, self.template, self_ns=self)

    @property
    def module(self):
        """The Python module referenced by this :class:`.Namespace`.

        If the namespace references a :class:`.Template`, then
        this module is the equivalent of ``template.module``,
        i.e. the generated module for the template.

        """
        return self.template.module

    @property
    def filename(self):
        """The path of the filesystem file used for this
        :class:`.Namespace`'s module or template.
        """
        return self.template.filename

    @property
    def uri(self):
        """The URI for this :class:`.Namespace`'s template.

        I.e. whatever was sent to :meth:`.TemplateLookup.get_template()`.

        This is the equivalent of :attr:`.Template.uri`.

        """
        return self.template.uri

    def _get_star(self):
        if self.callables:
            for key in self.callables:
                yield (key, self.callables[key])

        def get(key):
            callable_ = self.template._get_def_callable(key)
            return functools.partial(callable_, self.context)
        for k in self.template.module._exports:
            yield (k, get(k))

    def __getattr__(self, key):
        if key in self.callables:
            val = self.callables[key]
        elif self.template.has_def(key):
            callable_ = self.template._get_def_callable(key)
            val = functools.partial(callable_, self.context)
        elif self.inherits:
            val = getattr(self.inherits, key)
        else:
            raise AttributeError("Namespace '%s' has no member '%s'" % (self.name, key))
        setattr(self, key, val)
        return val