import re
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports
class BaseName:
    """
    The base class for all definitions, completions and signatures.
    """
    _mapping = {'posixpath': 'os.path', 'riscospath': 'os.path', 'ntpath': 'os.path', 'os2emxpath': 'os.path', 'macpath': 'os.path', 'genericpath': 'os.path', 'posix': 'os', '_io': 'io', '_functools': 'functools', '_collections': 'collections', '_socket': 'socket', '_sqlite3': 'sqlite3'}
    _tuple_mapping = dict(((tuple(k.split('.')), v) for k, v in {'argparse._ActionsContainer': 'argparse.ArgumentParser'}.items()))

    def __init__(self, inference_state, name):
        self._inference_state = inference_state
        self._name = name
        '\n        An instance of :class:`parso.python.tree.Name` subclass.\n        '
        self.is_keyword = isinstance(self._name, KeywordName)

    @memoize_method
    def _get_module_context(self):
        return self._name.get_root_context()

    @property
    def module_path(self) -> Optional[Path]:
        """
        Shows the file path of a module. e.g. ``/usr/lib/python3.9/os.py``
        """
        module = self._get_module_context()
        if module.is_stub() or not module.is_compiled():
            path: Optional[Path] = self._get_module_context().py__file__()
            return path
        return None

    @property
    def name(self):
        """
        Name of variable/function/class/module.

        For example, for ``x = None`` it returns ``'x'``.

        :rtype: str or None
        """
        return self._name.get_public_name()

    @property
    def type(self):
        """
        The type of the definition.

        Here is an example of the value of this attribute.  Let's consider
        the following source.  As what is in ``variable`` is unambiguous
        to Jedi, :meth:`jedi.Script.infer` should return a list of
        definition for ``sys``, ``f``, ``C`` and ``x``.

        >>> from jedi import Script
        >>> source = '''
        ... import keyword
        ...
        ... class C:
        ...     pass
        ...
        ... class D:
        ...     pass
        ...
        ... x = D()
        ...
        ... def f():
        ...     pass
        ...
        ... for variable in [keyword, f, C, x]:
        ...     variable'''

        >>> script = Script(source)
        >>> defs = script.infer()

        Before showing what is in ``defs``, let's sort it by :attr:`line`
        so that it is easy to relate the result to the source code.

        >>> defs = sorted(defs, key=lambda d: d.line)
        >>> print(defs)  # doctest: +NORMALIZE_WHITESPACE
        [<Name full_name='keyword', description='module keyword'>,
         <Name full_name='__main__.C', description='class C'>,
         <Name full_name='__main__.D', description='instance D'>,
         <Name full_name='__main__.f', description='def f'>]

        Finally, here is what you can get from :attr:`type`:

        >>> defs = [d.type for d in defs]
        >>> defs[0]
        'module'
        >>> defs[1]
        'class'
        >>> defs[2]
        'instance'
        >>> defs[3]
        'function'

        Valid values for type are ``module``, ``class``, ``instance``, ``function``,
        ``param``, ``path``, ``keyword``, ``property`` and ``statement``.

        """
        tree_name = self._name.tree_name
        resolve = False
        if tree_name is not None:
            definition = tree_name.get_definition()
            if definition is not None and definition.type == 'import_from' and tree_name.is_definition():
                resolve = True
        if isinstance(self._name, SubModuleName) or resolve:
            for value in self._name.infer():
                return value.api_type
        return self._name.api_type

    @property
    def module_name(self):
        """
        The module name, a bit similar to what ``__name__`` is in a random
        Python module.

        >>> from jedi import Script
        >>> source = 'import json'
        >>> script = Script(source, path='example.py')
        >>> d = script.infer()[0]
        >>> print(d.module_name)  # doctest: +ELLIPSIS
        json
        """
        return self._get_module_context().py__name__()

    def in_builtin_module(self):
        """
        Returns True, if this is a builtin module.
        """
        value = self._get_module_context().get_value()
        if isinstance(value, StubModuleValue):
            return any((v.is_compiled() for v in value.non_stub_value_set))
        return value.is_compiled()

    @property
    def line(self):
        """The line where the definition occurs (starting with 1)."""
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[0]

    @property
    def column(self):
        """The column where the definition occurs (starting with 0)."""
        start_pos = self._name.start_pos
        if start_pos is None:
            return None
        return start_pos[1]

    def get_definition_start_position(self):
        """
        The (row, column) of the start of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
        if self._name.tree_name is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.start_pos
        return definition.start_pos

    def get_definition_end_position(self):
        """
        The (row, column) of the end of the definition range. Rows start with
        1, columns start with 0.

        :rtype: Optional[Tuple[int, int]]
        """
        if self._name.tree_name is None:
            return None
        definition = self._name.tree_name.get_definition()
        if definition is None:
            return self._name.tree_name.end_pos
        if self.type in ('function', 'class'):
            last_leaf = definition.get_last_leaf()
            if last_leaf.type == 'newline':
                return last_leaf.get_previous_leaf().end_pos
            return last_leaf.end_pos
        return definition.end_pos

    def docstring(self, raw=False, fast=True):
        """
        Return a document string for this completion object.

        Example:

        >>> from jedi import Script
        >>> source = '''\\
        ... def f(a, b=1):
        ...     "Document for function f."
        ... '''
        >>> script = Script(source, path='example.py')
        >>> doc = script.infer(1, len('def f'))[0].docstring()
        >>> print(doc)
        f(a, b=1)
        <BLANKLINE>
        Document for function f.

        Notice that useful extra information is added to the actual
        docstring, e.g. function signatures are prepended to their docstrings.
        If you need the actual docstring, use ``raw=True`` instead.

        >>> print(script.infer(1, len('def f'))[0].docstring(raw=True))
        Document for function f.

        :param fast: Don't follow imports that are only one level deep like
            ``import foo``, but follow ``from foo import bar``. This makes
            sense for speed reasons. Completing `import a` is slow if you use
            the ``foo.docstring(fast=False)`` on every object, because it
            parses all libraries starting with ``a``.
        """
        if isinstance(self._name, ImportName) and fast:
            return ''
        doc = self._get_docstring()
        if raw:
            return doc
        signature_text = self._get_docstring_signature()
        if signature_text and doc:
            return signature_text + '\n\n' + doc
        else:
            return signature_text + doc

    def _get_docstring(self):
        return self._name.py__doc__()

    def _get_docstring_signature(self):
        return '\n'.join((signature.to_string() for signature in self._get_signatures(for_docstring=True)))

    @property
    def description(self):
        """
        A description of the :class:`.Name` object, which is heavily used
        in testing. e.g. for ``isinstance`` it returns ``def isinstance``.

        Example:

        >>> from jedi import Script
        >>> source = '''
        ... def f():
        ...     pass
        ...
        ... class C:
        ...     pass
        ...
        ... variable = f if random.choice([0,1]) else C'''
        >>> script = Script(source)  # line is maximum by default
        >>> defs = script.infer(column=3)
        >>> defs = sorted(defs, key=lambda d: d.line)
        >>> print(defs)  # doctest: +NORMALIZE_WHITESPACE
        [<Name full_name='__main__.f', description='def f'>,
         <Name full_name='__main__.C', description='class C'>]
        >>> str(defs[0].description)
        'def f'
        >>> str(defs[1].description)
        'class C'

        """
        typ = self.type
        tree_name = self._name.tree_name
        if typ == 'param':
            return typ + ' ' + self._name.to_string()
        if typ in ('function', 'class', 'module', 'instance') or tree_name is None:
            if typ == 'function':
                typ = 'def'
            return typ + ' ' + self._name.get_public_name()
        definition = tree_name.get_definition(include_setitem=True) or tree_name
        txt = definition.get_code(include_prefix=False)
        txt = re.sub('#[^\\n]+\\n', ' ', txt)
        txt = re.sub('\\s+', ' ', txt).strip()
        return txt

    @property
    def full_name(self):
        """
        Dot-separated path of this object.

        It is in the form of ``<module>[.<submodule>[...]][.<object>]``.
        It is useful when you want to look up Python manual of the
        object at hand.

        Example:

        >>> from jedi import Script
        >>> source = '''
        ... import os
        ... os.path.join'''
        >>> script = Script(source, path='example.py')
        >>> print(script.infer(3, len('os.path.join'))[0].full_name)
        os.path.join

        Notice that it returns ``'os.path.join'`` instead of (for example)
        ``'posixpath.join'``. This is not correct, since the modules name would
        be ``<module 'posixpath' ...>```. However most users find the latter
        more practical.
        """
        if not self._name.is_value_name:
            return None
        names = self._name.get_qualified_names(include_module_names=True)
        if names is None:
            return None
        names = list(names)
        try:
            names[0] = self._mapping[names[0]]
        except KeyError:
            pass
        return '.'.join(names)

    def is_stub(self):
        """
        Returns True if the current name is defined in a stub file.
        """
        if not self._name.is_value_name:
            return False
        return self._name.get_root_context().is_stub()

    def is_side_effect(self):
        """
        Checks if a name is defined as ``self.foo = 3``. In case of self, this
        function would return False, for foo it would return True.
        """
        tree_name = self._name.tree_name
        if tree_name is None:
            return False
        return tree_name.is_definition() and tree_name.parent.type == 'trailer'

    @debug.increase_indent_cm('goto on name')
    def goto(self, *, follow_imports=False, follow_builtin_imports=False, only_stubs=False, prefer_stubs=False):
        """
        Like :meth:`.Script.goto` (also supports the same params), but does it
        for the current name. This is typically useful if you are using
        something like :meth:`.Script.get_names()`.

        :param follow_imports: The goto call will follow imports.
        :param follow_builtin_imports: If follow_imports is True will try to
            look up names in builtins (i.e. compiled or extension modules).
        :param only_stubs: Only return stubs for this goto call.
        :param prefer_stubs: Prefer stubs to Python objects for this goto call.
        :rtype: list of :class:`Name`
        """
        if not self._name.is_value_name:
            return []
        names = self._name.goto()
        if follow_imports:
            names = filter_follow_imports(names, follow_builtin_imports)
        names = convert_names(names, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        return [self if n == self._name else Name(self._inference_state, n) for n in names]

    @debug.increase_indent_cm('infer on name')
    def infer(self, *, only_stubs=False, prefer_stubs=False):
        """
        Like :meth:`.Script.infer`, it can be useful to understand which type
        the current name has.

        Return the actual definitions. I strongly recommend not using it for
        your completions, because it might slow down |jedi|. If you want to
        read only a few objects (<=20), it might be useful, especially to get
        the original docstrings. The basic problem of this function is that it
        follows all results. This means with 1000 completions (e.g.  numpy),
        it's just very, very slow.

        :param only_stubs: Only return stubs for this goto call.
        :param prefer_stubs: Prefer stubs to Python objects for this type
            inference call.
        :rtype: list of :class:`Name`
        """
        assert not (only_stubs and prefer_stubs)
        if not self._name.is_value_name:
            return []
        names = convert_names([self._name], prefer_stubs=True)
        values = convert_values(ValueSet.from_sets((n.infer() for n in names)), only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        resulting_names = [c.name for c in values]
        return [self if n == self._name else Name(self._inference_state, n) for n in resulting_names]

    def parent(self):
        """
        Returns the parent scope of this identifier.

        :rtype: Name
        """
        if not self._name.is_value_name:
            return None
        if self.type in ('function', 'class', 'param') and self._name.tree_name is not None:
            cls_or_func_node = self._name.tree_name.get_definition()
            parent = search_ancestor(cls_or_func_node, 'funcdef', 'classdef', 'file_input')
            context = self._get_module_context().create_value(parent).as_context()
        else:
            context = self._name.parent_context
        if context is None:
            return None
        while context.name is None:
            context = context.parent_context
        return Name(self._inference_state, context.name)

    def __repr__(self):
        return '<%s %sname=%r, description=%r>' % (self.__class__.__name__, 'full_' if self.full_name else '', self.full_name or self.name, self.description)

    def get_line_code(self, before=0, after=0):
        """
        Returns the line of code where this object was defined.

        :param before: Add n lines before the current line to the output.
        :param after: Add n lines after the current line to the output.

        :return str: Returns the line(s) of code or an empty string if it's a
                     builtin.
        """
        if not self._name.is_value_name:
            return ''
        lines = self._name.get_root_context().code_lines
        if lines is None:
            return ''
        index = self._name.start_pos[0] - 1
        start_index = max(index - before, 0)
        return ''.join(lines[start_index:index + after + 1])

    def _get_signatures(self, for_docstring=False):
        if self._name.api_type == 'property':
            return []
        if for_docstring and self._name.api_type == 'statement' and (not self.is_stub()):
            return []
        if isinstance(self._name, MixedName):
            return self._name.infer_compiled_value().get_signatures()
        names = convert_names([self._name], prefer_stubs=True)
        return [sig for name in names for sig in name.infer().get_signatures()]

    def get_signatures(self):
        """
        Returns all potential signatures for a function or a class. Multiple
        signatures are typical if you use Python stubs with ``@overload``.

        :rtype: list of :class:`BaseSignature`
        """
        return [BaseSignature(self._inference_state, s) for s in self._get_signatures()]

    def execute(self):
        """
        Uses type inference to "execute" this identifier and returns the
        executed objects.

        :rtype: list of :class:`Name`
        """
        return _values_to_definitions(self._name.infer().execute_with_values())

    def get_type_hint(self):
        """
        Returns type hints like ``Iterable[int]`` or ``Union[int, str]``.

        This method might be quite slow, especially for functions. The problem
        is finding executions for those functions to return something like
        ``Callable[[int, str], str]``.

        :rtype: str
        """
        return self._name.infer().get_type_hint()