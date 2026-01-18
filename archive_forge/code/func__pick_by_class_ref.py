from saml2 import extension_elements_to_elements
from saml2.authn_context import ippword
from saml2.authn_context import mobiletwofactor
from saml2.authn_context import ppt
from saml2.authn_context import pword
from saml2.authn_context import sslcert
from saml2.saml import AuthnContext
from saml2.saml import AuthnContextClassRef
from saml2.samlp import RequestedAuthnContext
def _pick_by_class_ref(self, cls_ref, comparision_type='exact'):
    func = getattr(self, comparision_type)
    try:
        _refs = self.db['key'][cls_ref]
    except KeyError:
        return []
    else:
        _item = self.db['info'][_refs[0]]
        _level = _item['level']
        if comparision_type != 'better':
            if _item['method']:
                res = [(_item['method'], _refs[0])]
            else:
                res = []
        else:
            res = []
        for ref in _refs[1:]:
            item = self.db['info'][ref]
            res.append((item['method'], ref))
            if func(_level, item['level']):
                _level = item['level']
        for ref, _dic in self.db['info'].items():
            if ref in _refs:
                continue
            elif func(_level, _dic['level']):
                if _dic['method']:
                    _val = (_dic['method'], ref)
                    if _val not in res:
                        res.append(_val)
        return res