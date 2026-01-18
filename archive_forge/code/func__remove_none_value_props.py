import collections
from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
@staticmethod
def _remove_none_value_props(props):
    if isinstance(props, collections.abc.Mapping):
        return dict(((k, L2Gateway._remove_none_value_props(v)) for k, v in props.items() if v is not None))
    elif isinstance(props, collections.abc.Sequence) and (not isinstance(props, str)):
        return list((L2Gateway._remove_none_value_props(p) for p in props if p is not None))
    return props