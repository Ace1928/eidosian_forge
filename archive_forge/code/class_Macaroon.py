import abc
import base64
import json
import logging
import os
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery._utils import b64decode
from pymacaroons.serializers import json_serializer
from ._versions import (
from ._error import (
from ._codec import (
from ._keys import PublicKey
from ._third_party import (
class Macaroon(object):
    """Represent an undischarged macaroon along with its first
    party caveat namespace and associated third party caveat information
    which should be passed to the third party when discharging a caveat.
    """

    def __init__(self, root_key, id, location=None, version=LATEST_VERSION, namespace=None):
        """Creates a new macaroon with the given root key, id and location.

        If the version is more than the latest known version,
        the latest known version will be used. The namespace should hold the
        namespace of the service that is creating the macaroon.
        @param root_key bytes or string
        @param id bytes or string
        @param location bytes or string
        @param version the bakery version.
        @param namespace is that of the service creating it
        """
        if version > LATEST_VERSION:
            log.info('use last known version:{} instead of: {}'.format(LATEST_VERSION, version))
            version = LATEST_VERSION
        self._macaroon = pymacaroons.Macaroon(location=location, key=root_key, identifier=id, version=macaroon_version(version))
        self._version = version
        self._caveat_data = {}
        if namespace is None:
            namespace = checkers.Namespace()
        self._namespace = namespace
        self._caveat_id_prefix = bytearray()

    @property
    def macaroon(self):
        """ Return the underlying macaroon.
        """
        return self._macaroon

    @property
    def version(self):
        return self._version

    @property
    def namespace(self):
        return self._namespace

    @property
    def caveat_data(self):
        return self._caveat_data

    def add_caveat(self, cav, key=None, loc=None):
        """Add a caveat to the macaroon.

        It encrypts it using the given key pair
        and by looking up the location using the given locator.
        As a special case, if the caveat's Location field has the prefix
        "local " the caveat is added as a client self-discharge caveat using
        the public key base64-encoded in the rest of the location. In this
        case, the Condition field must be empty. The resulting third-party
        caveat will encode the condition "true" encrypted with that public
        key.

        @param cav the checkers.Caveat to be added.
        @param key the public key to encrypt third party caveat.
        @param loc locator to find information on third parties when adding
        third party caveats. It is expected to have a third_party_info method
        that will be called with a location string and should return a
        ThirdPartyInfo instance holding the requested information.
        """
        if cav.location is None:
            self._macaroon.add_first_party_caveat(self.namespace.resolve_caveat(cav).condition)
            return
        if key is None:
            raise ValueError('no private key to encrypt third party caveat')
        local_info = _parse_local_location(cav.location)
        if local_info is not None:
            if cav.condition:
                raise ValueError('cannot specify caveat condition in local third-party caveat')
            info = local_info
            cav = checkers.Caveat(location='local', condition='true')
        else:
            if loc is None:
                raise ValueError('no locator when adding third party caveat')
            info = loc.third_party_info(cav.location)
        root_key = os.urandom(24)
        if self._version < info.version:
            info = ThirdPartyInfo(version=self._version, public_key=info.public_key)
        caveat_info = encode_caveat(cav.condition, root_key, info, key, self._namespace)
        if info.version < VERSION_3:
            id = caveat_info
        else:
            id = self._new_caveat_id(self._caveat_id_prefix)
            self._caveat_data[id] = caveat_info
        self._macaroon.add_third_party_caveat(cav.location, root_key, id)

    def add_caveats(self, cavs, key, loc):
        """Add an array of caveats to the macaroon.

        This method does not mutate the current object.
        @param cavs arrary of caveats.
        @param key the PublicKey to encrypt third party caveat.
        @param loc locator to find the location object that has a method
        third_party_info.
        """
        if cavs is None:
            return
        for cav in cavs:
            self.add_caveat(cav, key, loc)

    def serialize_json(self):
        """Return a string holding the macaroon data in JSON format.
        @return a string holding the macaroon data in JSON format
        """
        return json.dumps(self.to_dict())

    def to_dict(self):
        """Return a dict representation of the macaroon data in JSON format.
        @return a dict
        """
        if self.version < VERSION_3:
            if len(self._caveat_data) > 0:
                raise ValueError('cannot serialize pre-version3 macaroon with external caveat data')
            return json.loads(self._macaroon.serialize(json_serializer.JsonSerializer()))
        serialized = {'m': json.loads(self._macaroon.serialize(json_serializer.JsonSerializer())), 'v': self._version}
        if self._namespace is not None:
            serialized['ns'] = self._namespace.serialize_text().decode('utf-8')
        caveat_data = {}
        for id in self._caveat_data:
            key = base64.b64encode(id).decode('utf-8')
            value = base64.b64encode(self._caveat_data[id]).decode('utf-8')
            caveat_data[key] = value
        if len(caveat_data) > 0:
            serialized['cdata'] = caveat_data
        return serialized

    @classmethod
    def from_dict(cls, json_dict):
        """Return a macaroon obtained from the given dictionary as
        deserialized from JSON.
        @param json_dict The deserialized JSON object.
        """
        json_macaroon = json_dict.get('m')
        if json_macaroon is None:
            m = pymacaroons.Macaroon.deserialize(json.dumps(json_dict), json_serializer.JsonSerializer())
            macaroon = Macaroon(root_key=None, id=None, namespace=legacy_namespace(), version=_bakery_version(m.version))
            macaroon._macaroon = m
            return macaroon
        version = json_dict.get('v', None)
        if version is None:
            raise ValueError('no version specified')
        if version < VERSION_3 or version > LATEST_VERSION:
            raise ValueError('unknown bakery version {}'.format(version))
        m = pymacaroons.Macaroon.deserialize(json.dumps(json_macaroon), json_serializer.JsonSerializer())
        if m.version != macaroon_version(version):
            raise ValueError('underlying macaroon has inconsistent version; got {} want {}'.format(m.version, macaroon_version(version)))
        namespace = checkers.deserialize_namespace(json_dict.get('ns'))
        cdata = json_dict.get('cdata', {})
        caveat_data = {}
        for id64 in cdata:
            id = b64decode(id64)
            data = b64decode(cdata[id64])
            caveat_data[id] = data
        macaroon = Macaroon(root_key=None, id=None, namespace=namespace, version=version)
        macaroon._caveat_data = caveat_data
        macaroon._macaroon = m
        return macaroon

    @classmethod
    def deserialize_json(cls, serialized_json):
        """Return a macaroon deserialized from a string
        @param serialized_json The string to decode {str}
        @return {Macaroon}
        """
        serialized = json.loads(serialized_json)
        return Macaroon.from_dict(serialized)

    def _new_caveat_id(self, base):
        """Return a third party caveat id

        This does not duplicate any third party caveat ids already inside
        macaroon. If base is non-empty, it is used as the id prefix.

        @param base bytes
        @return bytes
        """
        id = bytearray()
        if len(base) > 0:
            id.extend(base)
        else:
            id.append(VERSION_3)
        i = len(self._caveat_data)
        caveats = self._macaroon.caveats
        while True:
            temp = id[:]
            encode_uvarint(i, temp)
            found = False
            for cav in caveats:
                if cav.verification_key_id is not None and cav.caveat_id == temp:
                    found = True
                    break
            if not found:
                return bytes(temp)
            i += 1

    def first_party_caveats(self):
        """Return the first party caveats from this macaroon.

        @return the first party caveats from this macaroon as pymacaroons
        caveats.
        """
        return self._macaroon.first_party_caveats()

    def third_party_caveats(self):
        """Return the third party caveats.

        @return the third party caveats as pymacaroons caveats.
        """
        return self._macaroon.third_party_caveats()

    def copy(self):
        """ Returns a copy of the macaroon. Note that the the new
        macaroon's namespace still points to the same underlying Namespace -
        copying the macaroon does not make a copy of the namespace.
        :return a Macaroon
        """
        m1 = Macaroon(None, None, version=self._version, namespace=self._namespace)
        m1._macaroon = self._macaroon.copy()
        m1._caveat_data = self._caveat_data.copy()
        return m1