import base64
import datetime
import decimal
import json
from urllib.parse import urlencode
from wsme.exc import ClientSideError, InvalidInput
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.types import UserType, ArrayType, DictType
from wsme.rest import expose, validate
from wsme.rest.json import fromjson, tojson, parse
import wsme.tests.protocol
from wsme.utils import parse_isodatetime, parse_isotime, parse_isodate
class MiniCrud(object):

    @expose(CRUDResult, method='PUT')
    @validate(Obj)
    def create(self, data):
        print(repr(data))
        return CRUDResult(data, 'create')

    @expose(CRUDResult, method='GET', ignore_extra_args=True)
    @validate(Obj)
    def read(self, ref):
        print(repr(ref))
        if ref.id == 1:
            ref.name = 'test'
        return CRUDResult(ref, 'read')

    @expose(CRUDResult, method='POST')
    @validate(Obj)
    def update(self, data):
        print(repr(data))
        return CRUDResult(data, 'update')

    @expose(CRUDResult, wsme.types.text, body=Obj)
    def update_with_body(self, msg, data):
        print(repr(data))
        return CRUDResult(data, msg)

    @expose(CRUDResult, method='DELETE')
    @validate(Obj)
    def delete(self, ref):
        print(repr(ref))
        if ref.id == 1:
            ref.name = 'test'
        return CRUDResult(ref, 'delete')