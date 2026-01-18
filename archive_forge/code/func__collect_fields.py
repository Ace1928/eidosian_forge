import logging
from os_ken.services.protocols.bgp.operator.views import fields
@classmethod
def _collect_fields(cls):
    names = [attr for attr in dir(cls) if isinstance(getattr(cls, attr), fields.Field)]
    return dict([(name, getattr(cls, name)) for name in names])