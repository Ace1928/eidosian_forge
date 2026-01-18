import collections
import textwrap
from dataclasses import dataclass, field
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Callable
from xarray.core import duck_array_ops
from xarray.core.types import Dims, Self
@dataclass
class AggregationGenerator:
    _dim_docstring = _DIM_DOCSTRING
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE
    cls: str
    datastructure: DataStructure
    methods: tuple[Method, ...]
    docref: str
    docref_description: str
    example_call_preamble: str
    definition_preamble: str
    has_keep_attrs: bool = True
    notes: str = ''
    preamble: str = field(init=False)

    def __post_init__(self):
        self.preamble = self.definition_preamble.format(obj=self.datastructure.name, cls=self.cls)

    def generate_methods(self):
        yield [self.preamble]
        for method in self.methods:
            yield self.generate_method(method)

    def generate_method(self, method):
        has_kw_only = method.extra_kwargs or self.has_keep_attrs
        template_kwargs = dict(obj=self.datastructure.name, method=method.name, keep_attrs='\n        keep_attrs: bool | None = None,' if self.has_keep_attrs else '', kw_only='\n        *,' if has_kw_only else '')
        if method.extra_kwargs:
            extra_kwargs = '\n        ' + '\n        '.join([kwarg.kwarg for kwarg in method.extra_kwargs if kwarg.kwarg])
        else:
            extra_kwargs = ''
        yield self._template_signature.format(**template_kwargs, extra_kwargs=extra_kwargs)
        for text in [self._dim_docstring.format(method=method.name, cls=self.cls), *(kwarg.docs for kwarg in method.extra_kwargs if kwarg.docs), _KEEP_ATTRS_DOCSTRING if self.has_keep_attrs else None, _KWARGS_DOCSTRING.format(method=method.name)]:
            if text:
                yield textwrap.indent(text, 8 * ' ')
        yield TEMPLATE_RETURNS.format(**template_kwargs)
        others = self.datastructure.see_also_modules if self.cls == '' else (self.datastructure.name,)
        see_also_methods = '\n'.join((' ' * 8 + f'{mod}.{method.name}' for mod in method.see_also_modules + others))
        yield TEMPLATE_SEE_ALSO.format(**template_kwargs, docref=self.docref, docref_description=self.docref_description, see_also_methods=see_also_methods)
        notes = self.notes
        if method.numeric_only:
            if notes != '':
                notes += '\n\n'
            notes += _NUMERIC_ONLY_NOTES
        if notes != '':
            yield TEMPLATE_NOTES.format(notes=textwrap.indent(notes, 8 * ' '))
        yield textwrap.indent(self.generate_example(method=method), '')
        yield '        """'
        yield self.generate_code(method, self.has_keep_attrs)

    def generate_example(self, method):
        created = self.datastructure.create_example.format(example_array=method.np_example_array)
        calculation = f'{self.datastructure.example_var_name}{self.example_call_preamble}.{method.name}'
        if method.extra_kwargs:
            extra_examples = ''.join((kwarg.example for kwarg in method.extra_kwargs if kwarg.example)).format(calculation=calculation, method=method.name)
        else:
            extra_examples = ''
        return f'\n        Examples\n        --------{created}\n        >>> {self.datastructure.example_var_name}\n\n        >>> {calculation}(){extra_examples}'