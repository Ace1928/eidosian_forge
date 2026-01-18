from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
class OUI(BaseIdentifier):
    """
    An individual IEEE OUI (Organisationally Unique Identifier).

    For online details see - http://standards.ieee.org/regauth/oui/

    """
    __slots__ = ('records',)

    def __init__(self, oui):
        """
        Constructor

        :param oui: an OUI string ``XX-XX-XX`` or an unsigned integer.             Also accepts and parses full MAC/EUI-48 address strings (but not             MAC/EUI-48 integers)!
        """
        super(OUI, self).__init__()
        from netaddr.eui import ieee
        self.records = []
        if isinstance(oui, str):
            self._value = int(oui.replace('-', ''), 16)
        elif isinstance(oui, int):
            if 0 <= oui <= 16777215:
                self._value = oui
            else:
                raise ValueError('OUI int outside expected range: %r' % (oui,))
        else:
            raise TypeError('unexpected OUI format: %r' % (oui,))
        if self._value in ieee.OUI_INDEX:
            fh = _open_binary(__package__, 'oui.txt')
            for offset, size in ieee.OUI_INDEX[self._value]:
                fh.seek(offset)
                data = fh.read(size).decode('UTF-8')
                self._parse_data(data, offset, size)
            fh.close()
        else:
            raise NotRegisteredError('OUI %r not registered!' % (oui,))

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        if not isinstance(other, OUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return self._value == other._value

    def __ne__(self, other):
        if not isinstance(other, OUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return self._value != other._value

    def __getstate__(self):
        """:returns: Pickled state of an `OUI` object."""
        return (self._value, self.records)

    def __setstate__(self, state):
        """:param state: data used to unpickle a pickled `OUI` object."""
        self._value, self.records = state

    def _parse_data(self, data, offset, size):
        """Returns a dict record from raw OUI record data"""
        record = {'idx': 0, 'oui': '', 'org': '', 'address': [], 'offset': offset, 'size': size}
        for line in data.split('\n'):
            line = line.strip()
            if not line:
                continue
            if '(hex)' in line:
                record['idx'] = self._value
                record['org'] = line.split(None, 2)[2]
                record['oui'] = str(self)
            elif '(base 16)' in line:
                continue
            else:
                record['address'].append(line)
        self.records.append(record)

    @property
    def reg_count(self):
        """Number of registered organisations with this OUI"""
        return len(self.records)

    def registration(self, index=0):
        """
        The IEEE registration details for this OUI.

        :param index: the index of record (may contain multiple registrations)
            (Default: 0 - first registration)

        :return: Objectified Python data structure containing registration
            details.
        """
        return DictDotLookup(self.records[index])

    def __str__(self):
        """:return: string representation of this OUI"""
        int_val = self._value
        return '%02X-%02X-%02X' % (int_val >> 16 & 255, int_val >> 8 & 255, int_val & 255)

    def __repr__(self):
        """:return: executable Python string to recreate equivalent object."""
        return "OUI('%s')" % self