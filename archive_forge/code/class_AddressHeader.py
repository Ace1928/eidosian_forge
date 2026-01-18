from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class AddressHeader:
    max_count = None

    @staticmethod
    def value_parser(value):
        address_list, value = parser.get_address_list(value)
        assert not value, 'this should not happen'
        return address_list

    @classmethod
    def parse(cls, value, kwds):
        if isinstance(value, str):
            kwds['parse_tree'] = address_list = cls.value_parser(value)
            groups = []
            for addr in address_list.addresses:
                groups.append(Group(addr.display_name, [Address(mb.display_name or '', mb.local_part or '', mb.domain or '') for mb in addr.all_mailboxes]))
            defects = list(address_list.all_defects)
        else:
            if not hasattr(value, '__iter__'):
                value = [value]
            groups = [Group(None, [item]) if not hasattr(item, 'addresses') else item for item in value]
            defects = []
        kwds['groups'] = groups
        kwds['defects'] = defects
        kwds['decoded'] = ', '.join([str(item) for item in groups])
        if 'parse_tree' not in kwds:
            kwds['parse_tree'] = cls.value_parser(kwds['decoded'])

    def init(self, *args, **kw):
        self._groups = tuple(kw.pop('groups'))
        self._addresses = None
        super().init(*args, **kw)

    @property
    def groups(self):
        return self._groups

    @property
    def addresses(self):
        if self._addresses is None:
            self._addresses = tuple((address for group in self._groups for address in group.addresses))
        return self._addresses