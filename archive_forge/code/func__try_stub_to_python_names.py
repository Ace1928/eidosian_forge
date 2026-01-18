from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
@to_list
def _try_stub_to_python_names(names, prefer_stub_to_compiled=False):
    for name in names:
        module_context = name.get_root_context()
        if not module_context.is_stub():
            yield name
            continue
        if name.api_type == 'module':
            values = convert_values(name.infer(), ignore_compiled=prefer_stub_to_compiled)
            if values:
                for v in values:
                    yield v.name
                continue
        else:
            v = name.get_defining_qualified_value()
            if v is not None:
                converted = _stub_to_python_value_set(v, ignore_compiled=prefer_stub_to_compiled)
                if converted:
                    converted_names = converted.goto(name.get_public_name())
                    if converted_names:
                        for n in converted_names:
                            if n.get_root_context().is_stub():
                                yield name
                            else:
                                yield n
                        continue
        yield name