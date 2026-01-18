from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class Placement(dict):
    """
        Placement constraints to be used as part of a :py:class:`TaskTemplate`

        Args:
            constraints (:py:class:`list` of str): A list of constraints
            preferences (:py:class:`list` of tuple): Preferences provide a way
                to make the scheduler aware of factors such as topology. They
                are provided in order from highest to lowest precedence and
                are expressed as ``(strategy, descriptor)`` tuples. See
                :py:class:`PlacementPreference` for details.
            maxreplicas (int): Maximum number of replicas per node
            platforms (:py:class:`list` of tuple): A list of platforms
                expressed as ``(arch, os)`` tuples
    """

    def __init__(self, constraints=None, preferences=None, platforms=None, maxreplicas=None):
        if constraints is not None:
            self['Constraints'] = constraints
        if preferences is not None:
            self['Preferences'] = []
            for pref in preferences:
                if isinstance(pref, tuple):
                    pref = PlacementPreference(*pref)
                self['Preferences'].append(pref)
        if maxreplicas is not None:
            self['MaxReplicas'] = maxreplicas
        if platforms:
            self['Platforms'] = []
            for plat in platforms:
                self['Platforms'].append({'Architecture': plat[0], 'OS': plat[1]})