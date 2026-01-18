from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
class FakeIndirectionAPI(base.VersionedObjectIndirectionAPI):

    def __init__(self, serializer=None):
        super(FakeIndirectionAPI, self).__init__()
        self._ser = serializer or base.VersionedObjectSerializer()

    def _get_changes(self, orig_obj, new_obj):
        updates = dict()
        for name, field in new_obj.fields.items():
            if not new_obj.obj_attr_is_set(name):
                continue
            if not orig_obj.obj_attr_is_set(name) or getattr(orig_obj, name) != getattr(new_obj, name):
                updates[name] = field.to_primitive(new_obj, name, getattr(new_obj, name))
        return updates

    def _canonicalize_args(self, context, args, kwargs):
        args = tuple([self._ser.deserialize_entity(context, self._ser.serialize_entity(context, arg)) for arg in args])
        kwargs = dict([(argname, self._ser.deserialize_entity(context, self._ser.serialize_entity(context, arg))) for argname, arg in kwargs.items()])
        return (args, kwargs)

    def object_action(self, context, objinst, objmethod, args, kwargs):
        objinst = self._ser.deserialize_entity(context, self._ser.serialize_entity(context, objinst))
        objmethod = str(objmethod)
        args, kwargs = self._canonicalize_args(context, args, kwargs)
        original = objinst.obj_clone()
        with mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api', new=None):
            result = getattr(objinst, objmethod)(*args, **kwargs)
        updates = self._get_changes(original, objinst)
        updates['obj_what_changed'] = objinst.obj_what_changed()
        return (updates, result)

    def object_class_action(self, context, objname, objmethod, objver, args, kwargs):
        objname = str(objname)
        objmethod = str(objmethod)
        objver = str(objver)
        args, kwargs = self._canonicalize_args(context, args, kwargs)
        cls = base.VersionedObject.obj_class_from_name(objname, objver)
        with mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api', new=None):
            result = getattr(cls, objmethod)(context, *args, **kwargs)
        return base.VersionedObject.obj_from_primitive(result.obj_to_primitive(target_version=objver), context=context) if isinstance(result, base.VersionedObject) else result

    def object_class_action_versions(self, context, objname, objmethod, object_versions, args, kwargs):
        objname = str(objname)
        objmethod = str(objmethod)
        object_versions = {str(o): str(v) for o, v in object_versions.items()}
        args, kwargs = self._canonicalize_args(context, args, kwargs)
        objver = object_versions[objname]
        cls = base.VersionedObject.obj_class_from_name(objname, objver)
        with mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api', new=None):
            result = getattr(cls, objmethod)(context, *args, **kwargs)
        return base.VersionedObject.obj_from_primitive(result.obj_to_primitive(target_version=objver), context=context) if isinstance(result, base.VersionedObject) else result

    def object_backport(self, context, objinst, target_version):
        raise Exception('not supported')