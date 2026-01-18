import abc
from os_brick import executor
class VolumeEncryptor(executor.Executor, metaclass=abc.ABCMeta):
    """Base class to support encrypted volumes.

    A VolumeEncryptor provides hooks for attaching and detaching volumes, which
    are called immediately prior to attaching the volume to an instance and
    immediately following detaching the volume from an instance. This class
    performs no actions for either hook.
    """

    def __init__(self, root_helper, connection_info, keymgr, execute=None, *args, **kwargs):
        super(VolumeEncryptor, self).__init__(root_helper, *args, execute=execute, **kwargs)
        self._key_manager = keymgr
        self.encryption_key_id = kwargs.get('encryption_key_id')

    def _get_key(self, context):
        """Retrieves the encryption key for the specified volume.

        :param: the connection information used to attach the volume
        """
        return self._key_manager.get(context, self.encryption_key_id)

    @abc.abstractmethod
    def attach_volume(self, context, **kwargs):
        """Hook called immediately prior to attaching a volume to an instance.

        """
        pass

    @abc.abstractmethod
    def detach_volume(self, **kwargs):
        """Hook called immediately after detaching a volume from an instance.

        """
        pass

    @abc.abstractmethod
    def extend_volume(self, context, **kwargs):
        """Extend an encrypted volume and return the decrypted volume size."""
        pass