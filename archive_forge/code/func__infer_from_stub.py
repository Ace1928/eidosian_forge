from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
def _infer_from_stub(stub_module_context, qualified_names, ignore_compiled):
    from jedi.inference.compiled.mixed import MixedObject
    stub_module = stub_module_context.get_value()
    assert isinstance(stub_module, (StubModuleValue, MixedObject)), stub_module_context
    non_stubs = stub_module.non_stub_value_set
    if ignore_compiled:
        non_stubs = non_stubs.filter(lambda c: not c.is_compiled())
    for name in qualified_names:
        non_stubs = non_stubs.py__getattribute__(name)
    return non_stubs