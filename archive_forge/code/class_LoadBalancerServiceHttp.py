from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class LoadBalancerServiceHttp(BaseDomain):
    """LoadBalancerServiceHttp Domain

    :param cookie_name: str
        Name of the cookie used for Session Stickness
    :param cookie_lifetime: str
        Lifetime of the cookie used for Session Stickness
    :param certificates: list
            IDs of the Certificates to use for TLS/SSL termination by the Load Balancer; empty for TLS/SSL passthrough or if protocol is "http"
    :param redirect_http: bool
           Redirect traffic from http port 80 to port 443
    :param sticky_sessions: bool
           Use sticky sessions. Only available if protocol is "http" or "https".
    """

    def __init__(self, cookie_name: str | None=None, cookie_lifetime: str | None=None, certificates: list[BoundCertificate] | None=None, redirect_http: bool | None=None, sticky_sessions: bool | None=None):
        self.cookie_name = cookie_name
        self.cookie_lifetime = cookie_lifetime
        self.certificates = certificates
        self.redirect_http = redirect_http
        self.sticky_sessions = sticky_sessions