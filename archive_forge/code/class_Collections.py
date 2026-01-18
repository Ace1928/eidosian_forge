import enum
class Collections(enum.Enum):
    """Collections for all supported apis."""
    PROJECTS_LOCATIONS = ('projects.locations', '{+name}', {'': 'projects/{projectsId}/locations/{locationsId}'}, [u'name'])
    PROJECTS_LOCATIONS_KEYRINGS = ('projects.locations.keyRings', '{+name}', {'': 'projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}'}, [u'name'])
    PROJECTS_LOCATIONS_KEYRINGS_CRYPTOKEYS = ('projects.locations.keyRings.cryptoKeys', '{+name}', {'': 'projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}'}, [u'name'])
    PROJECTS_LOCATIONS_KEYRINGS_CRYPTOKEYS_CRYPTOKEYVERSIONS = ('projects.locations.keyRings.cryptoKeys.cryptoKeyVersions', '{+name}', {'': 'projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}'}, [u'name'])

    def __init__(self, collection_name, path, flat_paths, params):
        self.collection_name = collection_name
        self.path = path
        self.flat_paths = flat_paths
        self.params = params