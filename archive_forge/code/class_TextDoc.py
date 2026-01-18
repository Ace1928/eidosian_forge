import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
class TextDoc(Doc):
    """Formatter class for text documentation."""
    _repr_instance = TextRepr()
    repr = _repr_instance.repr

    def bold(self, text):
        """Format a string in bold by overstriking."""
        return ''.join((ch + '\x08' + ch for ch in text))

    def indent(self, text, prefix='    '):
        """Indent text by prepending a given prefix to each line."""
        if not text:
            return ''
        lines = [prefix + line for line in text.split('\n')]
        if lines:
            lines[-1] = lines[-1].rstrip()
        return '\n'.join(lines)

    def section(self, title, contents):
        """Format a section with a given heading."""
        clean_contents = self.indent(contents).rstrip()
        return self.bold(title) + '\n' + clean_contents + '\n\n'

    def formattree(self, tree, modname, parent=None, prefix=''):
        """Render in text a class tree as returned by inspect.getclasstree()."""
        result = ''
        for entry in tree:
            if type(entry) is type(()):
                c, bases = entry
                result = result + prefix + classname(c, modname)
                if bases and bases != (parent,):
                    parents = (classname(c, modname) for c in bases)
                    result = result + '(%s)' % ', '.join(parents)
                result = result + '\n'
            elif type(entry) is type([]):
                result = result + self.formattree(entry, modname, c, prefix + '    ')
        return result

    def docmodule(self, object, name=None, mod=None):
        """Produce text documentation for a given module object."""
        name = object.__name__
        synop, desc = splitdoc(getdoc(object))
        result = self.section('NAME', name + (synop and ' - ' + synop))
        all = getattr(object, '__all__', None)
        docloc = self.getdocloc(object)
        if docloc is not None:
            result = result + self.section('MODULE REFERENCE', docloc + '\n\nThe following documentation is automatically generated from the Python\nsource files.  It may be incomplete, incorrect or include features that\nare considered implementation detail and may vary between Python\nimplementations.  When in doubt, consult the module reference at the\nlocation listed above.\n')
        if desc:
            result = result + self.section('DESCRIPTION', desc)
        classes = []
        for key, value in inspect.getmembers(object, inspect.isclass):
            if all is not None or (inspect.getmodule(value) or object) is object:
                if visiblename(key, all, object):
                    classes.append((key, value))
        funcs = []
        for key, value in inspect.getmembers(object, inspect.isroutine):
            if all is not None or inspect.isbuiltin(value) or inspect.getmodule(value) is object:
                if visiblename(key, all, object):
                    funcs.append((key, value))
        data = []
        for key, value in inspect.getmembers(object, isdata):
            if visiblename(key, all, object):
                data.append((key, value))
        modpkgs = []
        modpkgs_names = set()
        if hasattr(object, '__path__'):
            for importer, modname, ispkg in pkgutil.iter_modules(object.__path__):
                modpkgs_names.add(modname)
                if ispkg:
                    modpkgs.append(modname + ' (package)')
                else:
                    modpkgs.append(modname)
            modpkgs.sort()
            result = result + self.section('PACKAGE CONTENTS', '\n'.join(modpkgs))
        submodules = []
        for key, value in inspect.getmembers(object, inspect.ismodule):
            if value.__name__.startswith(name + '.') and key not in modpkgs_names:
                submodules.append(key)
        if submodules:
            submodules.sort()
            result = result + self.section('SUBMODULES', '\n'.join(submodules))
        if classes:
            classlist = [value for key, value in classes]
            contents = [self.formattree(inspect.getclasstree(classlist, 1), name)]
            for key, value in classes:
                contents.append(self.document(value, key, name))
            result = result + self.section('CLASSES', '\n'.join(contents))
        if funcs:
            contents = []
            for key, value in funcs:
                contents.append(self.document(value, key, name))
            result = result + self.section('FUNCTIONS', '\n'.join(contents))
        if data:
            contents = []
            for key, value in data:
                contents.append(self.docother(value, key, name, maxlen=70))
            result = result + self.section('DATA', '\n'.join(contents))
        if hasattr(object, '__version__'):
            version = str(object.__version__)
            if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
                version = version[11:-1].strip()
            result = result + self.section('VERSION', version)
        if hasattr(object, '__date__'):
            result = result + self.section('DATE', str(object.__date__))
        if hasattr(object, '__author__'):
            result = result + self.section('AUTHOR', str(object.__author__))
        if hasattr(object, '__credits__'):
            result = result + self.section('CREDITS', str(object.__credits__))
        try:
            file = inspect.getabsfile(object)
        except TypeError:
            file = '(built-in)'
        result = result + self.section('FILE', file)
        return result

    def docclass(self, object, name=None, mod=None, *ignored):
        """Produce text documentation for a given class object."""
        realname = object.__name__
        name = name or realname
        bases = object.__bases__

        def makename(c, m=object.__module__):
            return classname(c, m)
        if name == realname:
            title = 'class ' + self.bold(realname)
        else:
            title = self.bold(name) + ' = class ' + realname
        if bases:
            parents = map(makename, bases)
            title = title + '(%s)' % ', '.join(parents)
        contents = []
        push = contents.append
        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != '()':
                push(name + argspec + '\n')
        doc = getdoc(object)
        if doc:
            push(doc + '\n')
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            push('Method resolution order:')
            for base in mro:
                push('    ' + makename(base))
            push('')
        subclasses = sorted((str(cls.__name__) for cls in type.__subclasses__(object) if not cls.__name__.startswith('_') and cls.__module__ == 'builtins'), key=str.lower)
        no_of_subclasses = len(subclasses)
        MAX_SUBCLASSES_TO_DISPLAY = 4
        if subclasses:
            push('Built-in subclasses:')
            for subclassname in subclasses[:MAX_SUBCLASSES_TO_DISPLAY]:
                push('    ' + subclassname)
            if no_of_subclasses > MAX_SUBCLASSES_TO_DISPLAY:
                push('    ... and ' + str(no_of_subclasses - MAX_SUBCLASSES_TO_DISPLAY) + ' other subclasses')
            push('')

        class HorizontalRule:

            def __init__(self):
                self.needone = 0

            def maybe(self):
                if self.needone:
                    push('-' * 70)
                self.needone = 1
        hr = HorizontalRule()

        def spill(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value, name, mod, object))
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    doc = getdoc(value)
                    try:
                        obj = getattr(object, name)
                    except AttributeError:
                        obj = homecls.__dict__[name]
                    push(self.docother(obj, name, mod, maxlen=70, doc=doc) + '\n')
            return attrs
        attrs = [(name, kind, cls, value) for name, kind, cls, value in classify_class_attrs(object) if visiblename(name, obj=object)]
        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            attrs, inherited = _split_list(attrs, lambda t: t[2] is thisclass)
            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = 'defined here'
            else:
                tag = 'inherited from %s' % classname(thisclass, object.__module__)
            sort_attributes(attrs, object)
            attrs = spill('Methods %s:\n' % tag, attrs, lambda t: t[1] == 'method')
            attrs = spill('Class methods %s:\n' % tag, attrs, lambda t: t[1] == 'class method')
            attrs = spill('Static methods %s:\n' % tag, attrs, lambda t: t[1] == 'static method')
            attrs = spilldescriptors('Readonly properties %s:\n' % tag, attrs, lambda t: t[1] == 'readonly property')
            attrs = spilldescriptors('Data descriptors %s:\n' % tag, attrs, lambda t: t[1] == 'data descriptor')
            attrs = spilldata('Data and other attributes %s:\n' % tag, attrs, lambda t: t[1] == 'data')
            assert attrs == []
            attrs = inherited
        contents = '\n'.join(contents)
        if not contents:
            return title + '\n'
        return title + '\n' + self.indent(contents.rstrip(), ' |  ') + '\n'

    def formatvalue(self, object):
        """Format an argument default value as text."""
        return '=' + self.repr(object)

    def docroutine(self, object, name=None, mod=None, cl=None):
        """Produce text documentation for a function or method object."""
        realname = object.__name__
        name = name or realname
        note = ''
        skipdocs = 0
        if _is_bound_method(object):
            imclass = object.__self__.__class__
            if cl:
                if imclass is not cl:
                    note = ' from ' + classname(imclass, mod)
            elif object.__self__ is not None:
                note = ' method of %s instance' % classname(object.__self__.__class__, mod)
            else:
                note = ' unbound %s method' % classname(imclass, mod)
        if inspect.iscoroutinefunction(object) or inspect.isasyncgenfunction(object):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''
        if name == realname:
            title = self.bold(realname)
        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                skipdocs = 1
            title = self.bold(name) + ' = ' + realname
        argspec = None
        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                argspec = str(signature)
                if realname == '<lambda>':
                    title = self.bold(name) + ' lambda '
                    argspec = argspec[1:-1]
        if not argspec:
            argspec = '(...)'
        decl = asyncqualifier + title + argspec + note
        if skipdocs:
            return decl + '\n'
        else:
            doc = getdoc(object) or ''
            return decl + '\n' + (doc and self.indent(doc).rstrip() + '\n')

    def docdata(self, object, name=None, mod=None, cl=None):
        """Produce text documentation for a data descriptor."""
        results = []
        push = results.append
        if name:
            push(self.bold(name))
            push('\n')
        doc = getdoc(object) or ''
        if doc:
            push(self.indent(doc))
            push('\n')
        return ''.join(results)
    docproperty = docdata

    def docother(self, object, name=None, mod=None, parent=None, maxlen=None, doc=None):
        """Produce text documentation for a data object."""
        repr = self.repr(object)
        if maxlen:
            line = (name and name + ' = ' or '') + repr
            chop = maxlen - len(line)
            if chop < 0:
                repr = repr[:chop] + '...'
        line = (name and self.bold(name) + ' = ' or '') + repr
        if not doc:
            doc = getdoc(object)
        if doc:
            line += '\n' + self.indent(str(doc)) + '\n'
        return line