import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def __sync_run(self) -> None:
    from time import sleep, time
    try:
        min_seconds, max_seconds = (10, 300)
        while self.__open:
            if self.__needs_sync:
                t0 = t1 = time()
                self.__needs_sync = False
                while self.__open:
                    sleep(0.1)
                    if self.__needs_sync:
                        t1 = time()
                        self.__needs_sync = False
                    if time() - t1 > min_seconds or time() - t0 > max_seconds:
                        self.__needs_sync = False
                        logger.debug('sync')
                        self.sync()
                        break
            else:
                sleep(1)
    except Exception as e:
        logger.exception(e)