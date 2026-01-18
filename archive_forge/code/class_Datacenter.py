from oslo_vmware._i18n import _
class Datacenter(object):

    def __init__(self, ref, name):
        """Datacenter object holds ref and name together for convenience."""
        if name is None:
            raise ValueError(_('Datacenter name cannot be None'))
        if ref is None:
            raise ValueError(_('Datacenter reference cannot be None'))
        self.ref = ref
        self.name = name