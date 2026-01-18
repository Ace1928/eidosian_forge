from pprint import pformat
from six import iteritems
import re
@access_modes.setter
def access_modes(self, access_modes):
    """
        Sets the access_modes of this V1PersistentVolumeClaimStatus.
        AccessModes contains the actual access modes the volume backing the PVC
        has. More info:
        https://kubernetes.io/docs/concepts/storage/persistent-volumes#access-modes-1

        :param access_modes: The access_modes of this
        V1PersistentVolumeClaimStatus.
        :type: list[str]
        """
    self._access_modes = access_modes