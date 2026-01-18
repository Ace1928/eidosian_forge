import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
class RRsetsReaderTransaction(dns.transaction.Transaction):

    def __init__(self, manager, replacement, read_only):
        assert not read_only
        super().__init__(manager, replacement, read_only)
        self.rdatasets = {}

    def _get_rdataset(self, name, rdtype, covers):
        return self.rdatasets.get((name, rdtype, covers))

    def _get_node(self, name):
        rdatasets = []
        for (rdataset_name, _, _), rdataset in self.rdatasets.items():
            if name == rdataset_name:
                rdatasets.append(rdataset)
        if len(rdatasets) == 0:
            return None
        node = dns.node.Node()
        node.rdatasets = rdatasets
        return node

    def _put_rdataset(self, name, rdataset):
        self.rdatasets[name, rdataset.rdtype, rdataset.covers] = rdataset

    def _delete_name(self, name):
        remove = []
        for key in self.rdatasets:
            if key[0] == name:
                remove.append(key)
        if len(remove) > 0:
            for key in remove:
                del self.rdatasets[key]

    def _delete_rdataset(self, name, rdtype, covers):
        try:
            del self.rdatasets[name, rdtype, covers]
        except KeyError:
            pass

    def _name_exists(self, name):
        for n, _, _ in self.rdatasets:
            if n == name:
                return True
        return False

    def _changed(self):
        return len(self.rdatasets) > 0

    def _end_transaction(self, commit):
        if commit and self._changed():
            rrsets = []
            for (name, _, _), rdataset in self.rdatasets.items():
                rrset = dns.rrset.RRset(name, rdataset.rdclass, rdataset.rdtype, rdataset.covers)
                rrset.update(rdataset)
                rrsets.append(rrset)
            self.manager.set_rrsets(rrsets)

    def _set_origin(self, origin):
        pass

    def _iterate_rdatasets(self):
        raise NotImplementedError

    def _iterate_names(self):
        raise NotImplementedError