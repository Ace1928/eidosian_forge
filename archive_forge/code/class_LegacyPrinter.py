import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
class LegacyPrinter:

    def display(self, _d: Union[str, Dict[str, Any]], **kwargs: Any) -> None:
        verbose = kwargs.get('verbose', False)
        padding = kwargs.get('padding', 0)
        obj: Optional[Union[Dict[str, Any], gitlab.base.RESTObject]] = kwargs.get('obj')
        if TYPE_CHECKING:
            assert obj is not None

        def display_dict(d: Dict[str, Any], padding: int) -> None:
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, dict):
                    print(f'{' ' * padding}{k.replace('_', '-')}:')
                    new_padding = padding + 2
                    self.display(v, verbose=True, padding=new_padding, obj=v)
                    continue
                print(f'{' ' * padding}{k.replace('_', '-')}: {v}')
        if verbose:
            if isinstance(obj, dict):
                display_dict(obj, padding)
                return
            if obj._id_attr:
                id = getattr(obj, obj._id_attr, None)
                print(f'{obj._id_attr}: {id}')
            attrs = obj.attributes
            if obj._id_attr:
                attrs.pop(obj._id_attr)
            display_dict(attrs, padding)
            return
        lines = []
        if TYPE_CHECKING:
            assert isinstance(obj, gitlab.base.RESTObject)
        if obj._id_attr:
            id = getattr(obj, obj._id_attr)
            lines.append(f'{obj._id_attr.replace('_', '-')}: {id}')
        if obj._repr_attr:
            value = getattr(obj, obj._repr_attr, 'None') or 'None'
            value = value.replace('\r', '').replace('\n', ' ')
            line = f'{obj._repr_attr}: {value}'
            if len(line) > 79:
                line = f'{line[:76]}...'
            lines.append(line)
        if lines:
            print('\n'.join(lines))
            return
        print(f"No default fields to show for {obj!r}. Please use  '--verbose' or the JSON/YAML formatters.")

    def display_list(self, data: List[Union[str, gitlab.base.RESTObject]], fields: List[str], **kwargs: Any) -> None:
        verbose = kwargs.get('verbose', False)
        for obj in data:
            if isinstance(obj, gitlab.base.RESTObject):
                self.display(get_dict(obj, fields), verbose=verbose, obj=obj)
            else:
                print(obj)
            print('')