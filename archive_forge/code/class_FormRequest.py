from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from lxml.html import (
from parsel.selector import create_root_node
from w3lib.html import strip_html5_whitespace
from scrapy.http.request import Request
from scrapy.http.response.text import TextResponse
from scrapy.utils.python import is_listlike, to_bytes
from scrapy.utils.response import get_base_url
class FormRequest(Request):
    valid_form_methods = ['GET', 'POST']

    def __init__(self, *args, formdata: FormdataType=None, **kwargs) -> None:
        if formdata and kwargs.get('method') is None:
            kwargs['method'] = 'POST'
        super().__init__(*args, **kwargs)
        if formdata:
            items = formdata.items() if isinstance(formdata, dict) else formdata
            form_query_str = _urlencode(items, self.encoding)
            if self.method == 'POST':
                self.headers.setdefault(b'Content-Type', b'application/x-www-form-urlencoded')
                self._set_body(form_query_str)
            else:
                self._set_url(urlunsplit(urlsplit(self.url)._replace(query=form_query_str)))

    @classmethod
    def from_response(cls: Type[FormRequestTypeVar], response: TextResponse, formname: Optional[str]=None, formid: Optional[str]=None, formnumber: int=0, formdata: FormdataType=None, clickdata: Optional[dict]=None, dont_click: bool=False, formxpath: Optional[str]=None, formcss: Optional[str]=None, **kwargs) -> FormRequestTypeVar:
        kwargs.setdefault('encoding', response.encoding)
        if formcss is not None:
            from parsel.csstranslator import HTMLTranslator
            formxpath = HTMLTranslator().css_to_xpath(formcss)
        form = _get_form(response, formname, formid, formnumber, formxpath)
        formdata = _get_inputs(form, formdata, dont_click, clickdata)
        url = _get_form_url(form, kwargs.pop('url', None))
        method = kwargs.pop('method', form.method)
        if method is not None:
            method = method.upper()
            if method not in cls.valid_form_methods:
                method = 'GET'
        return cls(url=url, method=method, formdata=formdata, **kwargs)