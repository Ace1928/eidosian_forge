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
class GroupByAggregationGenerator(AggregationGenerator):
    _dim_docstring = _DIM_DOCSTRING_GROUPBY
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE_GROUPBY

    def generate_code(self, method, has_keep_attrs):
        extra_kwargs = [kwarg.call for kwarg in method.extra_kwargs if kwarg.call]
        if self.datastructure.numeric_only:
            extra_kwargs.append(f'numeric_only={method.numeric_only},')
        method_is_not_flox_supported = method.name in ('median', 'cumsum', 'cumprod')
        if method_is_not_flox_supported:
            indent = 12
        else:
            indent = 16
        if extra_kwargs:
            extra_kwargs = textwrap.indent('\n' + '\n'.join(extra_kwargs), indent * ' ')
        else:
            extra_kwargs = ''
        if method_is_not_flox_supported:
            return f'        return self._reduce_without_squeeze_warn(\n            duck_array_ops.{method.array_method},\n            dim=dim,{extra_kwargs}\n            keep_attrs=keep_attrs,\n            **kwargs,\n        )'
        min_version_check = f'\n            and module_available("flox", minversion="{method.min_flox_version}")'
        return '        if (\n            flox_available\n            and OPTIONS["use_flox"]' + (min_version_check if method.min_flox_version is not None else '') + f'\n            and contains_only_chunked_or_numpy(self._obj)\n        ):\n            return self._flox_reduce(\n                func="{method.name}",\n                dim=dim,{extra_kwargs}\n                # fill_value=fill_value,\n                keep_attrs=keep_attrs,\n                **kwargs,\n            )\n        else:\n            return self._reduce_without_squeeze_warn(\n                duck_array_ops.{method.array_method},\n                dim=dim,{extra_kwargs}\n                keep_attrs=keep_attrs,\n                **kwargs,\n            )'