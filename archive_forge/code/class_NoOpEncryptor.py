from os_brick.encryptors import base
class NoOpEncryptor(base.VolumeEncryptor):
    """A VolumeEncryptor that does nothing.

    This class exists solely to wrap regular (i.e., unencrypted) volumes so
    that they do not require special handling with respect to an encrypted
    volume. This implementation performs no action when a volume is attached
    or detached.
    """

    def __init__(self, root_helper, connection_info, keymgr, execute=None, *args, **kwargs):
        super(NoOpEncryptor, self).__init__(*args, root_helper=root_helper, connection_info=connection_info, keymgr=keymgr, execute=execute, **kwargs)

    def attach_volume(self, context, **kwargs):
        pass

    def detach_volume(self, **kwargs):
        pass

    def extend_volume(self, context, **kwargs):
        pass