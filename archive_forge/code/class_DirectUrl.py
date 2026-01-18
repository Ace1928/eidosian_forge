import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
class DirectUrl:

    def __init__(self, url: str, info: InfoType, subdirectory: Optional[str]=None) -> None:
        self.url = url
        self.info = info
        self.subdirectory = subdirectory

    def _remove_auth_from_netloc(self, netloc: str) -> str:
        if '@' not in netloc:
            return netloc
        user_pass, netloc_no_user_pass = netloc.split('@', 1)
        if isinstance(self.info, VcsInfo) and self.info.vcs == 'git' and (user_pass == 'git'):
            return netloc
        if ENV_VAR_RE.match(user_pass):
            return netloc
        return netloc_no_user_pass

    @property
    def redacted_url(self) -> str:
        """url with user:password part removed unless it is formed with
        environment variables as specified in PEP 610, or it is ``git``
        in the case of a git URL.
        """
        purl = urllib.parse.urlsplit(self.url)
        netloc = self._remove_auth_from_netloc(purl.netloc)
        surl = urllib.parse.urlunsplit((purl.scheme, netloc, purl.path, purl.query, purl.fragment))
        return surl

    def validate(self) -> None:
        self.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DirectUrl':
        return DirectUrl(url=_get_required(d, str, 'url'), subdirectory=_get(d, str, 'subdirectory'), info=_exactly_one_of([ArchiveInfo._from_dict(_get(d, dict, 'archive_info')), DirInfo._from_dict(_get(d, dict, 'dir_info')), VcsInfo._from_dict(_get(d, dict, 'vcs_info'))]))

    def to_dict(self) -> Dict[str, Any]:
        res = _filter_none(url=self.redacted_url, subdirectory=self.subdirectory)
        res[self.info.name] = self.info._to_dict()
        return res

    @classmethod
    def from_json(cls, s: str) -> 'DirectUrl':
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def is_local_editable(self) -> bool:
        return isinstance(self.info, DirInfo) and self.info.editable