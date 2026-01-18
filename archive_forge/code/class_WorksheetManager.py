from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
class WorksheetManager(abc.ABC):
    """
    The Worksheet Manager
    """

    def __init__(self, adc: Optional[str]=None, url: Optional[str]=None, **kwargs):
        """
        The Worksheet Manager
        """
        self.adc = adc or os.getenv('GSPREAD_ADC', os.getenv('GOOGLE_APPLICATION_CREDENTIALS', None))
        self._gc: Optional[gspread.Client] = None
        self.default_url: Optional[str] = url
        self.url_to_ctx: Dict[str, str] = {}
        self.ctxs: Dict[str, WorksheetContext] = {}
        self._ctx: Optional[WorksheetContext] = None

    @property
    def gc(self) -> gspread.Client:
        """
        Returns the gspread client
        """
        if self._gc is None:
            if self.adc:
                self._gc = gspread.service_account(filename=self.adc)
            else:
                self._gc = gspread.oauth()
            self._gc.http_client.set_timeout(timeout=60 * 60)
        return self._gc

    @property
    def ctx(self) -> WorksheetContext:
        """
        The default context
        """
        if self._ctx is None:
            self.init_ctx()
        return self._ctx

    def init_ctx(self, name: Optional[str]=None, url: Optional[str]=None, set_as_default: Optional[bool]=False):
        """
        Initializes the context
        """
        name = name or 'default'
        if name not in self.ctxs:
            url = url or self.default_url
            assert url, 'No URL provided'
            self.ctxs[name] = WorksheetContext(self.gc, url)
            self.url_to_ctx[url] = name
        if self._ctx is None or set_as_default:
            self._ctx = self.ctxs[name]

    def get_ctx(self, name: Optional[str]=None, url: Optional[str]=None, set_as_default: Optional[bool]=False, **kwargs) -> WorksheetContext:
        """
        Returns a context
        """
        if name is None or name == 'default':
            return self.ctx
        self.init_ctx(name=name, url=url, set_as_default=set_as_default, **kwargs)
        return self.ctxs[name]