from __future__ import annotations
import ipaddress
import random
from typing import Any, Optional, Union
from pymongo.common import CONNECT_TIMEOUT
from pymongo.errors import ConfigurationError
class _SrvResolver:

    def __init__(self, fqdn: str, connect_timeout: Optional[float], srv_service_name: str, srv_max_hosts: int=0):
        self.__fqdn = fqdn
        self.__srv = srv_service_name
        self.__connect_timeout = connect_timeout or CONNECT_TIMEOUT
        self.__srv_max_hosts = srv_max_hosts or 0
        try:
            ipaddress.ip_address(fqdn)
            raise ConfigurationError(_INVALID_HOST_MSG % ('an IP address',))
        except ValueError:
            pass
        try:
            self.__plist = self.__fqdn.split('.')[1:]
        except Exception:
            raise ConfigurationError(_INVALID_HOST_MSG % (fqdn,)) from None
        self.__slen = len(self.__plist)
        if self.__slen < 2:
            raise ConfigurationError(_INVALID_HOST_MSG % (fqdn,))

    def get_options(self) -> Optional[str]:
        try:
            results = _resolve(self.__fqdn, 'TXT', lifetime=self.__connect_timeout)
        except (resolver.NoAnswer, resolver.NXDOMAIN):
            return None
        except Exception as exc:
            raise ConfigurationError(str(exc)) from None
        if len(results) > 1:
            raise ConfigurationError('Only one TXT record is supported')
        return b'&'.join([b''.join(res.strings) for res in results]).decode('utf-8')

    def _resolve_uri(self, encapsulate_errors: bool) -> resolver.Answer:
        try:
            results = _resolve('_' + self.__srv + '._tcp.' + self.__fqdn, 'SRV', lifetime=self.__connect_timeout)
        except Exception as exc:
            if not encapsulate_errors:
                raise
            raise ConfigurationError(str(exc)) from None
        return results

    def _get_srv_response_and_hosts(self, encapsulate_errors: bool) -> tuple[resolver.Answer, list[tuple[str, Any]]]:
        results = self._resolve_uri(encapsulate_errors)
        nodes = [(maybe_decode(res.target.to_text(omit_final_dot=True)), res.port) for res in results]
        for node in nodes:
            try:
                nlist = node[0].lower().split('.')[1:][-self.__slen:]
            except Exception:
                raise ConfigurationError(f'Invalid SRV host: {node[0]}') from None
            if self.__plist != nlist:
                raise ConfigurationError(f'Invalid SRV host: {node[0]}')
        if self.__srv_max_hosts:
            nodes = random.sample(nodes, min(self.__srv_max_hosts, len(nodes)))
        return (results, nodes)

    def get_hosts(self) -> list[tuple[str, Any]]:
        _, nodes = self._get_srv_response_and_hosts(True)
        return nodes

    def get_hosts_and_min_ttl(self) -> tuple[list[tuple[str, Any]], int]:
        results, nodes = self._get_srv_response_and_hosts(False)
        rrset = results.rrset
        ttl = rrset.ttl if rrset else 0
        return (nodes, ttl)