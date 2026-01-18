from __future__ import absolute_import, division, print_function
class Ospfv3Args(object):
    """The arg spec for the vyos_ospfv3 module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'areas': {'elements': 'dict', 'options': {'area_id': {'type': 'str'}, 'export_list': {'type': 'str'}, 'import_list': {'type': 'str'}, 'range': {'elements': 'dict', 'options': {'address': {'type': 'str'}, 'advertise': {'type': 'bool'}, 'not_advertise': {'type': 'bool'}}, 'type': 'list'}}, 'type': 'list'}, 'parameters': {'options': {'router_id': {'type': 'str'}}, 'type': 'dict'}, 'redistribute': {'elements': 'dict', 'options': {'route_map': {'type': 'str'}, 'route_type': {'choices': ['bgp', 'connected', 'kernel', 'ripng', 'static'], 'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'deleted', 'parsed', 'gathered', 'rendered'], 'default': 'merged', 'type': 'str'}}