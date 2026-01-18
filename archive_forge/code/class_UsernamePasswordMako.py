import logging
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from saml2 import SAMLError
import saml2.cryptography.symmetric
from saml2.httputil import Redirect
from saml2.httputil import Response
from saml2.httputil import Unauthorized
from saml2.httputil import make_cookie
from saml2.httputil import parse_cookie
class UsernamePasswordMako(UserAuthnMethod):
    """Do user authentication using the normal username password form
    using Mako as template system"""
    cookie_name = 'userpassmako'

    def __init__(self, srv, mako_template, template_lookup, pwd, return_to):
        """
        :param srv: The server instance
        :param mako_template: Which Mako template to use
        :param pwd: Username/password dictionary like database
        :param return_to: Where to send the user after authentication
        :return:
        """
        UserAuthnMethod.__init__(self, srv)
        self.mako_template = mako_template
        self.template_lookup = template_lookup
        self.passwd = pwd
        self.return_to = return_to
        self.active = {}
        self.query_param = 'upm_answer'
        self.symmetric = saml2.cryptography.symmetric.Default(srv.symkey)

    def __call__(self, cookie=None, policy_url=None, logo_url=None, query='', **kwargs):
        """
        Put up the login form
        """
        if cookie:
            headers = [cookie]
        else:
            headers = []
        resp = Response(headers=headers)
        argv = {'login': '', 'password': '', 'action': 'verify', 'policy_url': policy_url, 'logo_url': logo_url, 'query': query}
        logger.debug(f'do_authentication argv: {argv}')
        mte = self.template_lookup.get_template(self.mako_template)
        resp.message = mte.render(**argv)
        return resp

    def _verify(self, pwd, user):
        if not is_equal(pwd, self.passwd[user]):
            raise ValueError('Wrong password')

    def verify(self, request, **kwargs):
        """
        Verifies that the given username and password was correct
        :param request: Either the query part of a URL a urlencoded
            body of a HTTP message or a parse such.
        :param kwargs: Catch whatever else is sent.
        :return: redirect back to where ever the base applications
            wants the user after authentication.
        """
        if isinstance(request, str):
            _dict = parse_qs(request)
        elif isinstance(request, dict):
            _dict = request
        else:
            raise ValueError('Wrong type of input')
        try:
            self._verify(_dict['password'][0], _dict['login'][0])
            timestamp = str(int(time.mktime(time.gmtime())))
            msg = '::'.join([_dict['login'][0], timestamp])
            info = self.symmetric.encrypt(msg.encode())
            self.active[info] = timestamp
            cookie = make_cookie(self.cookie_name, info, self.srv.seed)
            return_to = create_return_url(self.return_to, _dict['query'][0], **{self.query_param: 'true'})
            resp = Redirect(return_to, headers=[cookie])
        except (ValueError, KeyError):
            resp = Unauthorized('Unknown user or wrong password')
        return resp

    def authenticated_as(self, cookie=None, **kwargs):
        if cookie is None:
            return None
        else:
            logger.debug(f'kwargs: {kwargs}')
            try:
                info, timestamp = parse_cookie(self.cookie_name, self.srv.seed, cookie)
                if self.active[info] == timestamp:
                    msg = self.symmetric.decrypt(info).decode()
                    uid, _ts = msg.split('::')
                    if timestamp == _ts:
                        return {'uid': uid}
            except Exception:
                pass
        return None

    def done(self, areq):
        try:
            _ = areq[self.query_param]
            return False
        except KeyError:
            return True