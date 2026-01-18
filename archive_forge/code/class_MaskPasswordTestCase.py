import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class MaskPasswordTestCase(test_base.BaseTestCase):

    def test_namespace_objects(self):
        payload = '\n        Namespace(passcode=\'\', username=\'\', password=\'my"password\',\n        profile=\'\', verify=None, token=\'\')\n        '
        expected = "\n        Namespace(passcode='', username='', password='***',\n        profile='', verify=None, token='***')\n        "
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_sanitize_keys(self):
        lowered = [k.lower() for k in strutils._SANITIZE_KEYS]
        message = 'The _SANITIZE_KEYS must all be lowercase.'
        self.assertEqual(strutils._SANITIZE_KEYS, lowered, message)

    def test_json(self):
        payload = "{'adminPass':'TL0EfN33'}"
        expected = "{'adminPass':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'adminPass' : 'TL0EfN33' }"
        expected = "{ 'adminPass' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'admin_pass':'TL0EfN33'}"
        expected = "{'admin_pass':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'admin_pass' : 'TL0EfN33' }"
        expected = "{ 'admin_pass' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'admin_password':'TL0EfN33'}"
        expected = "{'admin_password':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'admin_password' : 'TL0EfN33' }"
        expected = "{ 'admin_password' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'password':'TL0EfN33'}"
        expected = "{'password':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'password' : 'TL0EfN33' }"
        expected = "{ 'password' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'auth_password':'TL0EfN33'}"
        expected = "{'auth_password':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'auth_password' : 'TL0EfN33' }"
        expected = "{ 'auth_password' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'secret_uuid':'myuuid'}"
        expected = "{'secret_uuid':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'secret_uuid' : 'myuuid' }"
        expected = "{ 'secret_uuid' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'token':'token'}"
        expected = "{'token':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'token' : 'token' }"
        expected = "{ 'token' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'fernetkey' : 'token' }"
        expected = "{ 'fernetkey' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'FernetKey' : 'token' }"
        expected = "{ 'FernetKey' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'sslkey' : 'token' }"
        expected = "{ 'sslkey' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'SslKey' : 'token' }"
        expected = "{ 'SslKey' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'passphrase' : 'token' }"
        expected = "{ 'passphrase' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'PassPhrase' : 'token' }"
        expected = "{ 'PassPhrase' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'KeystoneFernetKey1' : 'token' }"
        expected = "{ 'KeystoneFernetKey1' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'OctaviaCaKeyPassword' : 'token' }"
        expected = "{ 'OctaviaCaKeyPassword' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{ 'OctaviaCaKeyPassphrase' : 'token' }"
        expected = "{ 'OctaviaCaKeyPassphrase' : '***' }"
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_xml(self):
        payload = '<adminPass>TL0EfN33</adminPass>'
        expected = '<adminPass>***</adminPass>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<adminPass>\n                        TL0EfN33\n                     </adminPass>'
        expected = '<adminPass>***</adminPass>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<admin_pass>TL0EfN33</admin_pass>'
        expected = '<admin_pass>***</admin_pass>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<admin_pass>\n                        TL0EfN33\n                     </admin_pass>'
        expected = '<admin_pass>***</admin_pass>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<admin_password>TL0EfN33</admin_password>'
        expected = '<admin_password>***</admin_password>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<admin_password>\n                        TL0EfN33\n                     </admin_password>'
        expected = '<admin_password>***</admin_password>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<password>TL0EfN33</password>'
        expected = '<password>***</password>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<password>\n                        TL0EfN33\n                     </password>'
        expected = '<password>***</password>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<Password1>TL0EfN33</Password1>'
        expected = '<Password1>***</Password1>'
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_xml_attribute(self):
        payload = "adminPass='TL0EfN33'"
        expected = "adminPass='***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "adminPass = 'TL0EfN33'"
        expected = "adminPass = '***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'adminPass = "TL0EfN33"'
        expected = 'adminPass = "***"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "admin_pass='TL0EfN33'"
        expected = "admin_pass='***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "admin_pass = 'TL0EfN33'"
        expected = "admin_pass = '***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'admin_pass = "TL0EfN33"'
        expected = 'admin_pass = "***"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "admin_password='TL0EfN33'"
        expected = "admin_password='***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "admin_password = 'TL0EfN33'"
        expected = "admin_password = '***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'admin_password = "TL0EfN33"'
        expected = 'admin_password = "***"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "password='TL0EfN33'"
        expected = "password='***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "password = 'TL0EfN33'"
        expected = "password = '***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'password = "TL0EfN33"'
        expected = 'password = "***"'
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_json_message(self):
        payload = 'body: {"changePassword": {"adminPass": "1234567"}}'
        expected = 'body: {"changePassword": {"adminPass": "***"}}'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'body: {"rescue": {"admin_pass": "1234567"}}'
        expected = 'body: {"rescue": {"admin_pass": "***"}}'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'body: {"rescue": {"admin_password": "1234567"}}'
        expected = 'body: {"rescue": {"admin_password": "***"}}'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'body: {"rescue": {"password": "1234567"}}'
        expected = 'body: {"rescue": {"password": "***"}}'
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_xml_message(self):
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rebuild\n    xmlns="http://docs.openstack.org/compute/api/v1.1"\n    name="foobar"\n    imageRef="http://openstack.example.com/v1.1/32278/images/70a599e0-31e7"\n    accessIPv4="1.2.3.4"\n    accessIPv6="fe80::100"\n    adminPass="seekr3t">\n  <metadata>\n    <meta key="My Server Name">Apache1</meta>\n  </metadata>\n</rebuild>'
        expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rebuild\n    xmlns="http://docs.openstack.org/compute/api/v1.1"\n    name="foobar"\n    imageRef="http://openstack.example.com/v1.1/32278/images/70a599e0-31e7"\n    accessIPv4="1.2.3.4"\n    accessIPv6="fe80::100"\n    adminPass="***">\n  <metadata>\n    <meta key="My Server Name">Apache1</meta>\n  </metadata>\n</rebuild>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_pass="MySecretPass"/>'
        expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_pass="***"/>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_password="MySecretPass"/>'
        expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    admin_password="***"/>'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    password="MySecretPass"/>'
        expected = '<?xml version="1.0" encoding="UTF-8"?>\n<rescue xmlns="http://docs.openstack.org/compute/api/v1.1"\n    password="***"/>'
        self.assertEqual(expected, strutils.mask_password(payload))

    def test_mask_password(self):
        payload = "test = 'password'  :   'aaaaaa'"
        expected = "test = 'password'  :   '111'"
        self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
        payload = 'mysqld --password "aaaaaa"'
        expected = 'mysqld --password "****"'
        self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
        payload = 'mysqld --password aaaaaa'
        expected = 'mysqld --password ???'
        self.assertEqual(expected, strutils.mask_password(payload, secret='???'))
        payload = 'mysqld --password = "aaaaaa"'
        expected = 'mysqld --password = "****"'
        self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
        payload = "mysqld --password = 'aaaaaa'"
        expected = "mysqld --password = '****'"
        self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
        payload = 'mysqld --password = aaaaaa'
        expected = 'mysqld --password = ****'
        self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
        payload = 'test = password =   aaaaaa'
        expected = 'test = password =   111'
        self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
        payload = 'test = password=   aaaaaa'
        expected = 'test = password=   111'
        self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
        payload = 'test = password =aaaaaa'
        expected = 'test = password =111'
        self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
        payload = 'test = password=aaaaaa'
        expected = 'test = password=111'
        self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
        payload = 'test = "original_password" : "aaaaaaaaa"'
        expected = 'test = "original_password" : "***"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = "param1" : "value"'
        expected = 'test = "param1" : "value"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = "original_password" : "aaaaa"aaaa"'
        expected = 'test = "original_password" : "***"'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'adminPass':'TL0EfN33'}"
        payload = str(payload)
        expected = "{'adminPass':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'adminPass':'TL0E'fN33'}"
        payload = str(payload)
        expected = "{'adminPass':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "{'token':'mytoken'}"
        payload = str(payload)
        expected = "{'token':'***'}"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "test = 'node.session.auth.password','-v','TL0EfN33','nomask'"
        expected = "test = 'node.session.auth.password','-v','***','nomask'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "test = 'node.session.auth.password', '--password', 'TL0EfN33', 'nomask'"
        expected = "test = 'node.session.auth.password', '--password', '***', 'nomask'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = "test = 'node.session.auth.password', '--password', 'TL0EfN33'"
        expected = "test = 'node.session.auth.password', '--password', '***'"
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = node.session.auth.password -v TL0EfN33 nomask'
        expected = 'test = node.session.auth.password -v *** nomask'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = node.session.auth.password --password TL0EfN33 nomask'
        expected = 'test = node.session.auth.password --password *** nomask'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = node.session.auth.password --password TL0EfN33'
        expected = 'test = node.session.auth.password --password ***'
        self.assertEqual(expected, strutils.mask_password(payload))
        payload = 'test = cmd --password myé\x80\x80pass'
        expected = 'test = cmd --password ***'
        self.assertEqual(expected, strutils.mask_password(payload))