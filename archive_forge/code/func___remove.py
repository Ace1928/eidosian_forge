import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def __remove(self, spo: Tuple[bytes, bytes, bytes], c: bytes, quoted: bool=False, txn: Optional[Any]=None) -> None:
    s, p, o = spo
    cspo, cpos, cosp = self.__indicies
    contexts_value = cspo.get('^'.encode('latin-1').join([''.encode('latin-1'), s, p, o, ''.encode('latin-1')]), txn=txn) or ''.encode('latin-1')
    contexts = set(contexts_value.split('^'.encode('latin-1')))
    contexts.discard(c)
    contexts_value = '^'.encode('latin-1').join(contexts)
    for i, _to_key, _from_key in self.__indicies_info:
        i.delete(_to_key((s, p, o), c), txn=txn)
    if not quoted:
        if contexts_value:
            for i, _to_key, _from_key in self.__indicies_info:
                i.put(_to_key((s, p, o), ''.encode('latin-1')), contexts_value, txn=txn)
        else:
            for i, _to_key, _from_key in self.__indicies_info:
                try:
                    i.delete(_to_key((s, p, o), ''.encode('latin-1')), txn=txn)
                except db.DBNotFoundError:
                    pass