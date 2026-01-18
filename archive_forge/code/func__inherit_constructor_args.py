from torch.testing._internal.opinfo.core import (
def _inherit_constructor_args(name, op, inherited, overrides):
    common_kwargs = {'name': name, 'op': op, 'aliases': None, 'method_variant': None, 'inplace_variant': None, 'supports_scripting': False}
    kwargs = inherited.copy()
    if 'kwargs' in kwargs:
        kwargs.update(kwargs['kwargs'])
        del kwargs['kwargs']
    if 'self' in kwargs:
        del kwargs['self']
    if '__class__' in kwargs:
        del kwargs['__class__']
    if 'skips' in kwargs:
        del kwargs['skips']
    if 'decorators' in kwargs:
        del kwargs['decorators']
    kwargs.update(common_kwargs)
    kwargs.update(overrides)
    kwargs['supports_autograd'] = False
    kwargs['supports_gradgrad'] = False
    kwargs['supports_fwgrad_bwgrad'] = False
    kwargs['supports_inplace_autograd'] = False
    kwargs['supports_forward_ad'] = False
    return kwargs