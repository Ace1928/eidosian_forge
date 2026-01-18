import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
from django.core.files import uploadhandler
from django.http.multipartparser import (
from django.utils.datastructures import (
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def _load_post_and_files(self):
    """Populate self._post and self._files if the content-type is a form type"""
    if self.method != 'POST':
        self._post, self._files = (QueryDict(encoding=self._encoding), MultiValueDict())
        return
    if self._read_started and (not hasattr(self, '_body')):
        self._mark_post_parse_error()
        return
    if self.content_type == 'multipart/form-data':
        if hasattr(self, '_body'):
            data = BytesIO(self._body)
        else:
            data = self
        try:
            self._post, self._files = self.parse_file_upload(self.META, data)
        except (MultiPartParserError, TooManyFilesSent):
            self._mark_post_parse_error()
            raise
    elif self.content_type == 'application/x-www-form-urlencoded':
        if self._encoding is not None and self._encoding.lower() != 'utf-8':
            raise BadRequest("HTTP requests with the 'application/x-www-form-urlencoded' content type must be UTF-8 encoded.")
        self._post = QueryDict(self.body, encoding='utf-8')
        self._files = MultiValueDict()
    else:
        self._post, self._files = (QueryDict(encoding=self._encoding), MultiValueDict())