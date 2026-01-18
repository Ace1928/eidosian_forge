import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def _init_db_environment(self, homeDir: str, create: bool=True) -> 'db.DBEnv':
    if not exists(homeDir):
        if create is True:
            mkdir(homeDir)
            self.create(homeDir)
        else:
            return NO_STORE
    db_env = db.DBEnv()
    db_env.set_cachesize(0, CACHESIZE)
    db_env.set_flags(ENVSETFLAGS, 1)
    db_env.open(homeDir, ENVFLAGS | db.DB_CREATE)
    return db_env